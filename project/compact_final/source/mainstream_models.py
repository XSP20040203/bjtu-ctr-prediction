"""DeepFM and parallel full-matrix DCN-v2 on the existing CTR features.

Paper architectures are adapted to the supplied short, unordered history lists:
each list is a masked-mean field and related IDs share an embedding vocabulary.
DeepFM uses shared FM/DNN embeddings, including scaled continuous fields.
DCN-v2 uses two dense cross layers, without the optional mixture-of-experts.
"""
import math

import torch
from torch import nn

from benchmark_core import masked_mean


def fm_interaction(fields):
    """Sum dot products over distinct field pairs, without self-interactions."""
    return .5 * (fields.sum(dim=1).square() - fields.square().sum(dim=1)).sum(dim=-1)


class CrossLayerV2(nn.Module):
    """x_(l+1) = x_0 * (W_l x_l + b_l) + x_l, using a full matrix."""
    def __init__(self, width):
        super().__init__()
        self.linear = nn.Linear(width, width)
        nn.init.xavier_normal_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, original, current):
        return original * self.linear(current) + current


class FieldEncoder(nn.Module):
    def __init__(self, categorical_fields, sequence_fields, group_for_field,
                 vocabulary_sizes, embedding_dim, unknown_dropout):
        super().__init__()
        self.categorical_fields = list(categorical_fields)
        self.sequence_fields = list(sequence_fields)
        self.group_for_field = dict(group_for_field)
        self.unknown_dropout = unknown_dropout
        self.embeddings = nn.ModuleDict({name: nn.Embedding(size, embedding_dim, padding_idx=0)
                                        for name, size in sorted(vocabulary_sizes.items())})
        for table in self.embeddings.values():
            nn.init.normal_(table.weight, mean=0., std=.02)
            with torch.no_grad():
                table.weight[0].zero_()

    def _indices(self, values, table):
        if self.training and self.unknown_dropout and table.num_embeddings > 100:
            replace = (values >= 2) & (torch.rand(values.shape, device=values.device) < self.unknown_dropout)
            return torch.where(replace, torch.ones_like(values), values)
        return values

    def forward(self, categories, sequences):
        fields, category_indices, sequence_indices = [], [], []
        for index, name in enumerate(self.categorical_fields):
            table = self.embeddings[self.group_for_field[name]]
            indices = self._indices(categories[:, index], table)
            category_indices.append(indices)
            fields.append(table(indices))
        for index, name in enumerate(self.sequence_fields):
            table = self.embeddings[self.group_for_field[name]]
            original = sequences[:, index]
            indices = self._indices(original, table)
            sequence_indices.append(indices)
            fields.append(masked_mean(table(indices), original.ne(0)))
        return torch.stack(fields, dim=1), category_indices, sequence_indices


def dense_network(width, hidden_dims, dropout):
    layers = []
    for index, hidden in enumerate(hidden_dims):
        layers.extend([nn.Linear(width, hidden), nn.ReLU()])
        if index < len(hidden_dims) - 1 and dropout:
            layers.append(nn.Dropout(dropout))
        width = hidden
    return nn.Sequential(*layers)


def initial_logit(initial_ctr):
    probability = .5 if initial_ctr is None else min(max(initial_ctr, 1e-6), 1 - 1e-6)
    return math.log(probability / (1 - probability))


class DeepFMModel(nn.Module):
    def __init__(self, categorical_fields, sequence_fields, group_for_field, vocabulary_sizes,
                 numeric_dim=20, embedding_dim=8, hidden_dims=(128, 64), dropout=.1,
                 unknown_dropout=.01, initial_ctr=None):
        super().__init__()
        self.encoder = FieldEncoder(categorical_fields, sequence_fields, group_for_field,
                                    vocabulary_sizes, embedding_dim, unknown_dropout)
        self.numeric_embeddings = nn.Parameter(torch.empty(numeric_dim, embedding_dim))
        nn.init.normal_(self.numeric_embeddings, mean=0., std=.02)
        # Linear coefficients are field-specific, even where FM/DNN ID tables are shared.
        self.linear_tables = nn.ModuleDict({
            name: nn.Embedding(vocabulary_sizes[group_for_field[name]], 1, padding_idx=0)
            for name in sorted(list(categorical_fields) + list(sequence_fields))})
        for table in self.linear_tables.values():
            nn.init.zeros_(table.weight)
        self.numeric_linear = nn.Linear(numeric_dim, 1, bias=False)
        nn.init.zeros_(self.numeric_linear.weight)
        field_count = len(categorical_fields) + len(sequence_fields) + numeric_dim
        self.deep = dense_network(field_count * embedding_dim, hidden_dims, dropout)
        self.deep_output = nn.Linear(hidden_dims[-1], 1, bias=False)
        self.bias = nn.Parameter(torch.tensor(initial_logit(initial_ctr)))

    def forward(self, categories, sequences, numeric):
        categorical_vectors, category_indices, sequence_indices = self.encoder(categories, sequences)
        numeric_vectors = numeric.unsqueeze(-1) * self.numeric_embeddings.unsqueeze(0)
        all_fields = torch.cat([categorical_vectors, numeric_vectors], dim=1)
        first_order = self.numeric_linear(numeric).squeeze(-1)
        for name, indices in zip(self.encoder.categorical_fields, category_indices):
            first_order = first_order + self.linear_tables[name](indices).squeeze(-1)
        for name, indices in zip(self.encoder.sequence_fields, sequence_indices):
            first_order = first_order + masked_mean(self.linear_tables[name](indices), indices.ne(0)).squeeze(-1)
        deep_logit = self.deep_output(self.deep(all_fields.flatten(start_dim=1))).squeeze(-1)
        return self.bias + first_order + fm_interaction(all_fields) + deep_logit


class DCNV2Model(nn.Module):
    def __init__(self, categorical_fields, sequence_fields, group_for_field, vocabulary_sizes,
                 numeric_dim=20, embedding_dim=8, hidden_dims=(128, 64), dropout=.1,
                 unknown_dropout=.01, initial_ctr=None, cross_layers=2):
        super().__init__()
        self.encoder = FieldEncoder(categorical_fields, sequence_fields, group_for_field,
                                    vocabulary_sizes, embedding_dim, unknown_dropout)
        width = (len(categorical_fields) + len(sequence_fields)) * embedding_dim + numeric_dim
        self.cross = nn.ModuleList([CrossLayerV2(width) for _ in range(cross_layers)])
        self.deep = dense_network(width, hidden_dims, dropout)
        self.output = nn.Linear(width + hidden_dims[-1], 1)
        nn.init.constant_(self.output.bias, initial_logit(initial_ctr))

    def forward(self, categories, sequences, numeric):
        fields, _, _ = self.encoder(categories, sequences)
        original = torch.cat([fields.flatten(start_dim=1), numeric], dim=-1)
        crossed = original
        for layer in self.cross:
            crossed = layer(original, crossed)
        return self.output(torch.cat([crossed, self.deep(original)], dim=-1)).squeeze(-1)


def mainstream_definition(name):
    options = {"embedding_dim": 8, "hidden_dims": [128, 64], "dropout": .1, "unknown_dropout": .01}
    if name == "deepfm":
        return DeepFMModel, options
    if name == "dcnv2":
        return DCNV2Model, {**options, "cross_layers": 2}
    if name == "mlp":
        from benchmark_core import CTRModel
        return CTRModel, options
    raise ValueError(f"Unknown architecture: {name}")
