"""Candidate-conditioned histories with an optional learned supplied-slot signal.

The backbone is official TabM; this input adapter is a custom experiment, not a
paper reproduction. Supplied list positions have no assumed time direction.
"""
import torch
from torch import nn

from modern_architectures import TabMFieldModel
from bounded_id_models import bound_field_vectors

QUERY_FIELDS = {
    f"ad_{action}_list_v00{number}": query
    for action in ["click", "close"]
    for number, query in enumerate(["task_id", "adv_prim_id", "spread_app_id"], 1)
}


class AdHistoryAttention(nn.Module):
    def __init__(self, dimension, hidden=16, max_slots=5, mode="position"):
        super().__init__()
        if mode not in ["mean", "query", "position"]:
            raise ValueError(f"Unknown history mode {mode}")
        self.mode, self.max_slots = mode, max_slots
        self.score = nn.Sequential(nn.Linear(dimension * 4, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.score[2].weight)
        nn.init.zeros_(self.score[2].bias)
        self.match_weight = nn.Parameter(torch.zeros(()))
        self.position_bias = nn.Parameter(torch.zeros(max_slots))
        self.position_match = nn.Parameter(torch.zeros(max_slots))
        self.match_values = nn.Parameter(torch.zeros(max_slots, dimension))

    def forward(self, history, query, mask, exact_match):
        if history.shape[1] > self.max_slots:
            raise ValueError("History exceeds the configured supplied-slot length")
        length = history.shape[1]
        values, expanded = history.float(), query.float()[:, None].expand_as(history)
        if self.mode == "mean":
            return ((values * mask[..., None]).sum(dim=1) /
                    mask.sum(dim=1, keepdim=True).clamp_min(1)).to(history.dtype)
        interaction = torch.cat([values, expanded, values - expanded, values * expanded], dim=-1)
        matches = (exact_match & mask).float()
        scores = self.score(interaction).squeeze(-1).float() + self.match_weight * matches
        if self.mode == "position":
            scores = scores + self.position_bias[:length] + self.position_match[:length] * matches
        scores = scores.masked_fill(~mask, torch.finfo(torch.float32).min)
        weights = scores.softmax(dim=-1) * mask.float()
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        pooled = (values * weights[..., None]).sum(dim=1)
        if self.mode == "position":
            pooled = pooled + (matches[..., None] * self.match_values[None, :length]).sum(dim=1)
        return pooled.to(history.dtype)


class AdHistoryTabM(TabMFieldModel):
    def __init__(self, *args, history_mode="position", history_max_slots=5, **kwargs):
        super().__init__(*args, **kwargs)
        self.history_mode = history_mode
        dimension = next(iter(self.encoder.embeddings.values())).embedding_dim
        fields = [name for name in self.encoder.sequence_fields if name in QUERY_FIELDS]
        for name in fields:
            query = QUERY_FIELDS[name]
            if query not in self.encoder.categorical_fields or self.encoder.group_for_field[query] != self.encoder.group_for_field[name]:
                raise ValueError(f"History and its query must share an ID vocabulary: {name}")
        self.history_attention = nn.ModuleDict({name: AdHistoryAttention(dimension, max_slots=history_max_slots, mode=history_mode)
                                               for name in fields})
        # Mean, query and position controls allocate the same parameters and use
        # the same neutral initialization; only the information path differs.
        self.history_mix_logits = nn.Parameter(torch.zeros(len(fields)))

    def forward(self, categories, sequences, numeric):
        fields, category_indices, sequence_indices = self.encoder(categories, sequences)
        pieces = list(fields.unbind(dim=1))
        category_count = len(self.encoder.categorical_fields)
        for parameter_index, (name, attention) in enumerate(self.history_attention.items()):
            sequence_column = self.encoder.sequence_fields.index(name)
            query_column = self.encoder.categorical_fields.index(QUERY_FIELDS[name])
            query_codes, history_codes = category_indices[query_column], sequence_indices[sequence_column]
            # UNK equality and padding equality do not count as item matches.
            exact_match = (history_codes == query_codes[:, None]) & (history_codes >= 2) & (query_codes[:, None] >= 2)
            history = self.encoder.embeddings[self.encoder.group_for_field[name]](history_codes)
            selected = attention(history, fields[:, query_column], sequences[:, sequence_column].ne(0), exact_match)
            index = category_count + sequence_column
            gate = torch.sigmoid(self.history_mix_logits[parameter_index])
            pieces[index] = pieces[index] + gate * (selected - pieces[index])
        fields = bound_field_vectors(torch.stack(pieces, dim=1), self.field_groups, self.group_norm_caps)
        dense = numeric if self.numeric_embeddings is None else self.numeric_embeddings(numeric).flatten(1)
        features = torch.cat([fields.flatten(1), dense], dim=-1)
        hidden = self.backbone(features[:, None].expand(-1, self.k, -1))
        return self.head(hidden).squeeze(-1)
