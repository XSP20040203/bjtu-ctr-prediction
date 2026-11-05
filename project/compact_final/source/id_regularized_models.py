"""DCN-v2 with configurable ID-group dropout and unchanged default initialization."""
import torch
from torch import nn

from mainstream_models import FieldEncoder, CrossLayerV2, dense_network, initial_logit


class GroupDropoutEncoder(FieldEncoder):
    def __init__(self, *args, group_dropout_rates=None, **kwargs):
        super().__init__(*args, **kwargs)
        rates = dict(group_dropout_rates or {})
        for name, probability in rates.items():
            if name not in self.embeddings or not 0 <= probability <= 1:
                raise ValueError(f"Invalid ID dropout override: {name}={probability}")
        # Modules retain identity when moved to CUDA; checkpoints recreate this map.
        self._rate_for_table = {table: rates.get(name, self.unknown_dropout)
                                for name, table in self.embeddings.items()}

    def _indices(self, values, table):
        probability = self._rate_for_table[table]
        if self.training and probability and table.num_embeddings > 100:
            replace = (values >= 2) & (torch.rand(values.shape, device=values.device) < probability)
            return torch.where(replace, torch.ones_like(values), values)
        return values


class IDRegularizedDCN(nn.Module):
    def __init__(self, categorical_fields, sequence_fields, group_for_field, vocabulary_sizes,
                 numeric_dim=20, embedding_dim=8, hidden_dims=(128, 64), dropout=.1,
                 unknown_dropout=.01, initial_ctr=None, cross_layers=2, group_dropout_rates=None):
        super().__init__()
        self.encoder = GroupDropoutEncoder(categorical_fields, sequence_fields, group_for_field,
                                          vocabulary_sizes, embedding_dim, unknown_dropout,
                                          group_dropout_rates=group_dropout_rates)
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
