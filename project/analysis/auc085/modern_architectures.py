"""Modern tabular backbones adapted to categorical IDs and short CTR histories.

TabM uses the authors' tabm package. FieldTransformer is a compact field-token
attention comparison, not a reproduction of a paper's reported benchmark.
"""
import math
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
import tabm
import rtdl_num_embeddings

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "model_comparison"))
from id_regularized_models import GroupDropoutEncoder
from bounded_id_models import bound_field_vectors


def fit_numeric_bins(values, n_bins=12):
    values = np.asarray(values)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Numeric bin fitting requires a finite matrix of training rows")
    result = []
    for column in values.T:
        edges = np.unique(np.quantile(column, np.linspace(0, 1, n_bins + 1)).astype(np.float32))
        if len(edges) < 2:
            center = float(edges[0])
            epsilon = max(1., abs(center) * .01)
            edges = np.array([center - epsilon, center + epsilon], dtype=np.float32)
        result.append(edges.tolist())
    return result


def member_probability(logits):
    return torch.sigmoid(logits.float()).mean(dim=1)


class TabMFieldModel(nn.Module):
    def __init__(self, categorical_fields, sequence_fields, group_for_field, vocabulary_sizes,
                 numeric_dim=75, embedding_dim=8, n_blocks=3, d_block=256, k=8,
                 dropout=.1, unknown_dropout=.01, initial_ctr=None, group_norm_caps=None,
                 numeric_bin_edges=None, numeric_embedding_dim=8):
        super().__init__()
        self.k = k
        self.encoder = GroupDropoutEncoder(categorical_fields, sequence_fields, group_for_field,
                                          vocabulary_sizes, embedding_dim, unknown_dropout)
        self.field_groups = [group_for_field[name] for name in list(categorical_fields) + list(sequence_fields)]
        self.group_norm_caps = dict({"user_id": .2, "ad_task": .2, "adv_id": .2} if group_norm_caps is None else group_norm_caps)
        self.numeric_embeddings = None
        numeric_width = numeric_dim
        if numeric_bin_edges is not None:
            if len(numeric_bin_edges) != numeric_dim:
                raise ValueError("One bin list is required for every numeric feature")
            self.numeric_embeddings = rtdl_num_embeddings.PiecewiseLinearEmbeddings(
                [torch.tensor(edges, dtype=torch.float32) for edges in numeric_bin_edges],
                d_embedding=numeric_embedding_dim, activation=False, version="B")
            numeric_width = numeric_dim * numeric_embedding_dim
        field_count = len(categorical_fields) + len(sequence_fields)
        chunks = [embedding_dim] * field_count + [numeric_embedding_dim if self.numeric_embeddings is not None else 1] * numeric_dim
        self.backbone = tabm.make_tabm_backbone(d_in=field_count * embedding_dim + numeric_width,
                                               n_blocks=n_blocks, d_block=d_block, dropout=dropout,
                                               k=k, arch_type="tabm", start_scaling_init="normal",
                                               start_scaling_init_chunks=chunks)
        self.head = tabm.LinearEnsemble(d_block, 1, k=k)
        if initial_ctr is not None:
            probability = min(max(initial_ctr, 1e-6), 1 - 1e-6)
            nn.init.constant_(self.head.bias, math.log(probability / (1 - probability)))

    def forward(self, categories, sequences, numeric):
        fields, _, _ = self.encoder(categories, sequences)
        fields = bound_field_vectors(fields, self.field_groups, self.group_norm_caps)
        dense = numeric if self.numeric_embeddings is None else self.numeric_embeddings(numeric).flatten(1)
        features = torch.cat([fields.flatten(1), dense], dim=-1)
        hidden = self.backbone(features[:, None].expand(-1, self.k, -1))
        return self.head(hidden).squeeze(-1)


class ClippedTabMFieldModel(TabMFieldModel):
    def __init__(self, *args, numeric_bin_edges, **kwargs):
        super().__init__(*args, numeric_bin_edges=numeric_bin_edges, **kwargs)
        self.register_buffer("numeric_lower", torch.tensor([edges[0] for edges in numeric_bin_edges], dtype=torch.float32))
        self.register_buffer("numeric_upper", torch.tensor([edges[-1] for edges in numeric_bin_edges], dtype=torch.float32))

    def forward(self, categories, sequences, numeric):
        bounded = numeric.clamp(min=self.numeric_lower, max=self.numeric_upper)
        return super().forward(categories, sequences, bounded)


class FieldTransformerModel(nn.Module):
    def __init__(self, categorical_fields, sequence_fields, group_for_field, vocabulary_sizes,
                 numeric_dim=75, embedding_dim=16, d_token=32, n_layers=2, n_heads=4,
                 dropout=.1, unknown_dropout=.01, initial_ctr=None):
        super().__init__()
        self.encoder = GroupDropoutEncoder(categorical_fields, sequence_fields, group_for_field,
                                          vocabulary_sizes, embedding_dim, unknown_dropout,
                                          group_dropout_rates={"user_id": .2, "ad_task": .1, "adv_id": .1})
        self.category_projection = nn.Linear(embedding_dim, d_token)
        self.numeric_weight = nn.Parameter(torch.empty(numeric_dim, d_token))
        self.numeric_bias = nn.Parameter(torch.zeros(numeric_dim, d_token))
        self.cls = nn.Parameter(torch.empty(1, 1, d_token))
        count = len(categorical_fields) + len(sequence_fields) + numeric_dim + 1
        self.field_identity = nn.Parameter(torch.empty(1, count, d_token))
        for value in [self.numeric_weight, self.cls, self.field_identity]:
            nn.init.normal_(value, std=.02)
        layer = nn.TransformerEncoderLayer(d_model=d_token, nhead=n_heads, dim_feedforward=d_token * 4,
                                          dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.output_norm = nn.LayerNorm(d_token)
        self.head = nn.Linear(d_token, 1)
        if initial_ctr is not None:
            probability = min(max(initial_ctr, 1e-6), 1 - 1e-6)
            nn.init.constant_(self.head.bias, math.log(probability / (1 - probability)))

    def forward(self, categories, sequences, numeric):
        fields, _, _ = self.encoder(categories, sequences)
        categorical_tokens = self.category_projection(fields)
        numeric_tokens = numeric[..., None] * self.numeric_weight + self.numeric_bias
        tokens = torch.cat([self.cls.expand(len(categories), -1, -1), categorical_tokens, numeric_tokens], dim=1)
        encoded = self.transformer(tokens + self.field_identity)
        return self.head(self.output_norm(encoded[:, 0]))
