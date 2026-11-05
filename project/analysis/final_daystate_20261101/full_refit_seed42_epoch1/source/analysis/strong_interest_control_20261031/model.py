"""One controlled pooling change on the original strong GatedCross backbone."""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "architecture_goal_20261029"))
import torch
from torch import nn
from interaction_models import GatedCrossCTR, CappedInteractionEncoder

NEWS_FIELDS = ["u_newsCatInterestsST", "feed_long_interest", "feed_dislike_topics",
               "feed_clicked_topics", "feed_click_ca2", "feed_exposed_topics"]
QUERY_FIELDS = ["adv_prim_id", "spread_app_id", "app_second_class", "slot_id"]


class InterestEncoder(CappedInteractionEncoder):
    def install_attention(self):
        self.query_indices = [self.categorical_fields.index(name) for name in QUERY_FIELDS]
        self.interest_scores = nn.ModuleDict()
        # Preserve the backbone's CPU and CUDA initialization/random streams.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(20261031)
            self.interest_query = nn.Linear(len(QUERY_FIELDS) * self.embedding_dim,
                                           self.embedding_dim)
            for name in NEWS_FIELDS:
                self.interest_scores[name] = nn.Sequential(
                    nn.Linear(4 * self.embedding_dim, 16), nn.SiLU(), nn.Linear(16, 1))
                nn.init.zeros_(self.interest_scores[name][-1].weight)
                nn.init.zeros_(self.interest_scores[name][-1].bias)

    def forward(self, categories, sequences):
        fields, category_indices, sequence_indices = [], [], []
        for index, name in enumerate(self.categorical_fields):
            vectors, indices = self.read(name, categories[:, index])
            fields.append(vectors)
            category_indices.append(indices)
        query = torch.tanh(self.interest_query(torch.cat(
            [torch.tanh(fields[index] / .02) for index in self.query_indices], dim=-1)))
        for index, name in enumerate(self.sequence_fields):
            original = sequences[:, index]
            vectors, indices = self.read(name, original)
            mask = original.ne(0)
            weights = mask.unsqueeze(-1).to(vectors.dtype)
            mean = (vectors * weights).sum(1) / weights.sum(1).clamp_min(1.)
            pooled = mean
            if name in self.interest_scores:
                key = torch.tanh(vectors / .02)
                expanded_query = query.unsqueeze(1).expand_as(key)
                interaction = torch.cat([expanded_query, key, expanded_query * key,
                                         expanded_query - key], dim=-1)
                scores = self.interest_scores[name](interaction).squeeze(-1)
                scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
                attention = scores.softmax(-1) * mask.to(scores.dtype)
                attention = attention / attention.sum(-1, keepdim=True).clamp_min(1e-12)
                attentive = (vectors * attention.unsqueeze(-1)).sum(1)
                count = mask.sum(-1, keepdim=True).to(vectors.dtype)
                # Sparse/singleton feedback preserves the original pooled field.
                support = count / (count + 3.)
                pooled = mean + support * (attentive - mean)
            fields.append(pooled)
            sequence_indices.append(indices)
        return torch.stack(fields, dim=1), category_indices, sequence_indices


def build(meta, mode, prior):
    model = GatedCrossCTR(meta["categorical_fields"], meta["sequence_fields"],
                          meta["group_for_field"], meta["vocabulary_sizes"],
                          numeric_dim=166, initial_ctr=prior)
    if mode == "attention":
        model.encoder.__class__ = InterestEncoder
        model.encoder.install_attention()
    elif mode != "control":
        raise ValueError("Unknown controlled arm")
    return model
