"""TabM with source-graph user interests and explicit ad-query gated matching."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from modern_architectures import TabMFieldModel
from bounded_id_models import bound_field_vectors


class GraphInterestTabM(TabMFieldModel):
    def __init__(self, *args, graph_dim=32, protected_source=True, **kwargs):
        super().__init__(*args, **kwargs)
        user_table = self.encoder.embeddings["user_id"]
        user_table.weight.requires_grad_(False)  # Replaced, not silently retained as an ID shortcut.
        self.source_user_vectors = nn.Parameter(torch.zeros(user_table.num_embeddings, graph_dim),
                                                requires_grad=not protected_source)
        self.register_buffer("source_positive_count", torch.zeros(user_table.num_embeddings))
        self.user_projection = nn.Linear(graph_dim, user_table.embedding_dim, bias=False)
        self.item_query = nn.Sequential(nn.Linear(user_table.embedding_dim * 4, graph_dim, bias=False), nn.LayerNorm(graph_dim))
        self.graph_gate = nn.Linear(kwargs.get("d_block", 256) + graph_dim, 1)
        nn.init.zeros_(self.graph_gate.weight)
        nn.init.constant_(self.graph_gate.bias, -2.)
        self.graph_score_scale = nn.Parameter(torch.zeros(()))
        self.confidence_temperature = nn.Parameter(torch.tensor(math.log(math.expm1(3.))))
        self.user_column = self.encoder.categorical_fields.index("user_id")

    def forward(self, categories, sequences, numeric):
        fields, category_indices, _ = self.encoder(categories, sequences)
        codes = category_indices[self.user_column]
        source = F.normalize(self.source_user_vectors[codes].float(), dim=-1, eps=1e-8)
        count = self.source_positive_count[codes]
        confidence = count / (count + F.softplus(self.confidence_temperature) + 1e-8)
        projected_user = self.user_projection(source) * confidence[:, None]
        pieces = list(fields.unbind(dim=1))
        pieces[self.user_column] = projected_user
        fields = bound_field_vectors(torch.stack(pieces, dim=1), self.field_groups, self.group_norm_caps)
        lookup = dict(zip(self.encoder.categorical_fields + self.encoder.sequence_fields, fields.unbind(dim=1)))
        zero = torch.zeros_like(fields[:, 0])
        item = torch.cat([lookup.get(name, zero) for name in ["task_id", "spread_app_id", "adv_prim_id", "app_second_class"]], dim=-1)
        query = F.normalize(self.item_query(item).float(), dim=-1, eps=1e-8)
        match = (source * query).sum(dim=-1) * confidence
        dense = numeric if self.numeric_embeddings is None else self.numeric_embeddings(numeric).flatten(1)
        features = torch.cat([fields.flatten(1), dense], dim=-1)
        hidden = self.backbone(features[:, None].expand(-1, self.k, -1))
        expanded_source = source[:, None].expand(-1, self.k, -1)
        gate = torch.sigmoid(self.graph_gate(torch.cat([hidden, expanded_source], dim=-1))).squeeze(-1)
        return self.head(hidden).squeeze(-1).float() + self.graph_score_scale * gate.float() * match[:, None]
