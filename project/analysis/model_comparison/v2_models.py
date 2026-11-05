"""Controlled CTR architecture variants, reusing the baseline's feature contract."""
import math

import torch
from torch import nn

from benchmark_core import CTRModel, masked_mean

QUERIES = {"ad_click_list_v001": "task_id", "ad_click_list_v002": "adv_prim_id", "ad_click_list_v003": "spread_app_id"}


class TargetAttention(nn.Module):
    def __init__(self, embedding_dim=8, hidden_dim=16):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(embedding_dim * 4, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, history, query, mask):
        expanded_query = query.unsqueeze(1).expand_as(history)
        interaction = torch.cat([history, expanded_query, history - expanded_query, history * expanded_query], dim=-1)
        scores = self.score(interaction).squeeze(-1)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1) * mask.to(scores.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        return (history * weights.unsqueeze(-1)).sum(dim=1)


class OptimizedCTRModel(CTRModel):
    def __init__(self, categorical_fields, sequence_fields, group_for_field, vocabulary_sizes,
                 numeric_dim=20, embedding_dim=8, hidden_dims=(128, 64), dropout=.1,
                 unknown_dropout=.01, initial_ctr=None, attention=True, source_gate=False):
        super().__init__(categorical_fields, sequence_fields, group_for_field, vocabulary_sizes,
                         numeric_dim, embedding_dim, hidden_dims, dropout, unknown_dropout, initial_ctr)
        self.use_source_gate = source_gate
        self.attentions = nn.ModuleDict()
        if attention:
            for name in sorted(set(sequence_fields).intersection(QUERIES)):
                query_name = QUERIES[name]
                if query_name not in categorical_fields or group_for_field[query_name] != group_for_field[name]:
                    raise ValueError(f"Attention query does not share the correct ID space: {name}")
                self.attentions[name] = TargetAttention(embedding_dim, 16)
        if source_gate:
            if numeric_dim != 20:
                raise ValueError("Source gate expects 13 target numeric features followed by 7 source features")
            del self.mlp
            target_width = (len(categorical_fields) + len(sequence_fields)) * embedding_dim + 13
            self.target_network = nn.Sequential(nn.Linear(target_width, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, 64), nn.ReLU())
            self.source_network = nn.Sequential(nn.Linear(7, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU())
            self.source_gate = nn.Linear(80, 16)
            self.prediction_head = nn.Sequential(nn.Linear(80, 32), nn.ReLU(), nn.Linear(32, 1))
            if initial_ctr is not None:
                probability = min(max(initial_ctr, 1e-6), 1 - 1e-6)
                nn.init.constant_(self.prediction_head[-1].bias, math.log(probability / (1 - probability)))

    def forward(self, categories, sequences, numeric):
        category_vectors = {}
        for index, name in enumerate(self.categorical_fields):
            table = self.embeddings[self.group_for_field[name]]
            category_vectors[name] = table(self._drop_unknown(categories[:, index], table))
        pieces = [category_vectors[name] for name in self.categorical_fields]
        for index, name in enumerate(self.sequence_fields):
            table = self.embeddings[self.group_for_field[name]]
            original = sequences[:, index]
            vectors = table(self._drop_unknown(original, table))
            mask = original.ne(0)
            if name in self.attentions:
                pooled = self.attentions[name](vectors, category_vectors[QUERIES[name]], mask)
            else:
                pooled = masked_mean(vectors, mask)
            pieces.append(pooled)
        if self.use_source_gate:
            target = self.target_network(torch.cat(pieces + [numeric[:, :13]], dim=-1))
            source = self.source_network(numeric[:, 13:])
            weight = torch.sigmoid(self.source_gate(torch.cat([target, source], dim=-1)))
            return self.prediction_head(torch.cat([target, source * weight], dim=-1)).squeeze(-1)
        return self.mlp(torch.cat(pieces + [numeric], dim=-1)).squeeze(-1)


def variant_definition(name):
    options = {"embedding_dim": 8, "hidden_dims": [128, 64], "dropout": .1, "unknown_dropout": .01}
    if name == "baseline":
        return CTRModel, {**options, "embedding_dim": 16, "hidden_dims": [256, 128, 64]}
    if name == "B1":
        return CTRModel, {**options, "hidden_dims": [256, 128, 64]}
    if name == "B2":
        return CTRModel, options
    if name in ["B3", "B4"]:
        return OptimizedCTRModel, {**options, "attention": True, "source_gate": name == "B4"}
    raise ValueError(f"Unknown variant: {name}")
