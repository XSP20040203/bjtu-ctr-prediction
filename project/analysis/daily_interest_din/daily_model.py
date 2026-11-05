"""Candidate-conditioned interest over unordered daily topic distributions."""
from pathlib import Path
import math
import sys

import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "model_comparison"))
from bounded_id_models import BoundedIDDCN


class DailyInterestDCN(BoundedIDDCN):
    def __init__(self, categorical_fields, sequence_fields, group_for_field, vocabulary_sizes,
                 n_topics, context_dim, interest_mode="control", time_mode="both",
                 interest_dim=8, interest_hidden=32, **options):
        super().__init__(categorical_fields, sequence_fields, group_for_field, vocabulary_sizes, **options)
        if interest_mode not in {"control", "mean", "din"}:
            raise ValueError("Unknown interest mode")
        if time_mode not in {"past", "today", "both"}:
            raise ValueError("Unknown time scope")
        if n_topics < 2 or context_dim < 1:
            raise ValueError("Topic vocabulary and context are required")
        self.interest_mode, self.time_mode = interest_mode, time_mode
        self.n_topics, self.context_dim, self.interest_dim = n_topics, context_dim, interest_dim
        self.query_fields = ["spread_app_id", "app_second_class", "adv_prim_id"]
        self.query_columns = [categorical_fields.index(field) for field in self.query_fields]
        if interest_mode == "control":
            return
        # Model-specific initialization must not perturb the shared backbone random stream.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(314159)
            self.query_embeddings = nn.ModuleList([nn.Embedding(vocabulary_sizes[group_for_field[field]], interest_dim, padding_idx=0)
                                                   for field in self.query_fields])
            self.topic_embedding = nn.Embedding(n_topics, interest_dim, padding_idx=0)
            for table in [*self.query_embeddings, self.topic_embedding]:
                nn.init.normal_(table.weight, std=.05)
                with torch.no_grad():
                    table.weight[0].zero_()
            self.query_projection = nn.Linear(len(self.query_fields)*interest_dim, interest_dim)
            if interest_mode == "din":
                self.activation = nn.Sequential(nn.Linear(interest_dim*4, interest_hidden), nn.PReLU(),
                                                nn.Linear(interest_hidden, 1))
                nn.init.zeros_(self.activation[-1].weight)
                nn.init.zeros_(self.activation[-1].bias)
            pooled_width = 2*4*interest_dim
            self.interest_head = nn.Sequential(nn.Linear(pooled_width + interest_dim + 2*context_dim, interest_hidden),
                                               nn.SiLU(), nn.Linear(interest_hidden, 1))
            nn.init.zeros_(self.interest_head[-1].weight)
            nn.init.zeros_(self.interest_head[-1].bias)

    def candidate_query(self, categories):
        return self.query_projection(torch.cat([table(categories[:, column])
                for table, column in zip(self.query_embeddings, self.query_columns)], dim=-1))

    def interest_readout(self, categories, weights, context):
        if weights.ndim != 4 or weights.shape[1:] != (2, 4, self.n_topics):
            raise ValueError("Expected [batch, past/today, four roles, topic] normalized counts")
        if context.shape[1:] != (2, self.context_dim):
            raise ValueError("Unexpected daily context dimensions")
        weights, context = weights.float(), context.float()
        mask = weights.new_tensor([1., 1.] if self.time_mode == "both" else
                                  [1., 0.] if self.time_mode == "past" else [0., 1.])
        weights = weights * mask[None, :, None, None]
        context = context * mask[None, :, None]
        # PAD/UNK have no verified semantics; explicit masses/missing flags remain in context.
        weights = weights.clone()
        weights[..., :2] = 0
        query = self.candidate_query(categories)
        topics = self.topic_embedding.weight
        if self.interest_mode == "din":
            q = query[:, None].expand(-1, self.n_topics, -1)
            t = topics[None].expand(len(query), -1, -1)
            local = torch.cat([q, t, q-t, q*t], dim=-1)
            activation = 2*torch.sigmoid(self.activation(local).squeeze(-1))
            weights = weights * activation[:, None, None, :]
        pooled = torch.einsum("bdrt,te->bdre", weights, topics)
        return pooled, query, context

    def forward(self, categories, sequences, numeric, topic_weights, interest_context):
        base = super().forward(categories, sequences, numeric)
        if self.interest_mode == "control":
            return base
        pooled, query, context = self.interest_readout(categories, topic_weights, interest_context)
        features = torch.cat([pooled.flatten(1), query, context.flatten(1)], dim=-1)
        return base + self.interest_head(features).squeeze(-1)
