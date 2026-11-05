"""Item-conditioned interest transfer shared between news and advertisement tasks."""
import torch
from torch import nn

from cross_domain_model import CrossDomainGatedTabM
from bounded_id_models import bound_field_vectors


class SharedInterestAttention(nn.Module):
    def __init__(self, dimension, hidden=32):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(dimension * 4, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, history, query, mask):
        values, query32 = history.float(), query.float()
        expanded = query32[:, None].expand_as(values)
        interaction = torch.cat([values, expanded, values - expanded, values * expanded], dim=-1)
        scores = self.score(interaction).squeeze(-1).float()
        scores = scores.masked_fill(~mask, torch.finfo(torch.float32).min)
        weights = scores.softmax(dim=-1) * mask.float()
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return (values * weights[..., None]).sum(dim=1).to(history.dtype)


class AttentiveCrossDomainTabM(CrossDomainGatedTabM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        dimension = self.encoder.embeddings["user_id"].embedding_dim
        self.interest_attention = SharedInterestAttention(dimension)
        self.ad_query = nn.Sequential(nn.Linear(dimension * 4, dimension), nn.LayerNorm(dimension))
        self.source_query = nn.Sequential(nn.Linear(dimension, dimension), nn.LayerNorm(dimension))
        nn.init.zeros_(self.ad_query[0].bias)
        nn.init.zeros_(self.source_query[0].bias)
        nn.init.eye_(self.source_query[0].weight)

    def _ad_profile(self, lookup, sequences, sequence_indices):
        zero = torch.zeros_like(lookup["user_id"])
        item = torch.cat([lookup.get(name, zero) for name in ["task_id", "spread_app_id", "adv_prim_id", "app_second_class"]], dim=-1)
        query = self.ad_query(item)
        interests = []
        for name in ["feed_long_interest", "u_newsCatInterestsST", "feed_click_ca2"]:
            if name not in self.encoder.sequence_fields:
                interests.append(zero)
                continue
            index = self.encoder.sequence_fields.index(name)
            table = self.encoder.embeddings[self.encoder.group_for_field[name]]
            vectors = table(sequence_indices[index])
            interests.append(self.interest_attention(vectors, query, sequences[:, index].ne(0)))
        return self.profile_tower(torch.cat([lookup["user_id"]] + interests, dim=-1))

    def _source_profile(self, vectors, history_vectors, sequences):
        query = self.source_query(vectors["i_cat"])
        user = bound_field_vectors(vectors["u_userId"][:, None], ["user_id"], self.group_norm_caps)[:, 0]
        interests = []
        for name in self.source_profile_sequences:
            index = self.source_schema["sequence_fields"].index(name)
            interests.append(self.interest_attention(history_vectors[name], query, sequences[:, index].ne(0)))
        return self.profile_tower(torch.cat([user] + interests, dim=-1))
