"""TabM with a shared user-interest tower and gated cross-domain transfer."""
import math

import torch
from torch import nn

from modern_architectures import TabMFieldModel
from benchmark_core import masked_mean
from bounded_id_models import bound_field_vectors


class CrossDomainGatedTabM(TabMFieldModel):
    def __init__(self, *args, source_schema, profile_dim=64, **kwargs):
        forbidden = {"label", "pro", "cillabel"}
        names = source_schema["categorical_fields"] + source_schema["sequence_fields"] + source_schema["numeric_fields"]
        if forbidden.intersection(name.lower() for name in names):
            raise ValueError("Source-click inputs cannot contain the response label or post-click progress/like response")
        super().__init__(*args, **kwargs)
        self.source_schema = source_schema
        dimension = next(iter(self.encoder.embeddings.values())).embedding_dim
        width = kwargs.get("d_block", 256)
        self.profile_tower = nn.Sequential(nn.Linear(dimension * 4, profile_dim), nn.ReLU(),
                                           nn.Linear(profile_dim, profile_dim), nn.LayerNorm(profile_dim))
        self.transfer_projection = nn.Linear(profile_dim, width, bias=False)
        nn.init.zeros_(self.transfer_projection.weight)
        self.transfer_gate = nn.Linear(width + profile_dim, width)
        nn.init.constant_(self.transfer_gate.bias, -2.)
        self.source_embeddings = nn.ModuleDict()
        for group, size in sorted(source_schema["vocabulary_sizes"].items()):
            if group in self.encoder.embeddings:
                if self.encoder.embeddings[group].num_embeddings != size:
                    raise ValueError(f"Shared source vocabulary mismatch: {group}")
            else:
                table = nn.Embedding(size, dimension, padding_idx=0)
                nn.init.normal_(table.weight, std=.02)
                with torch.no_grad():
                    table.weight[0].zero_()
                self.source_embeddings[group] = table
        # User and three positive-interest histories go through the shared tower.
        self.source_profile_sequences = ["u_newsCatInterests", "u_newsCatInterestsST", "u_click_ca2_news"]
        extra_count = len(source_schema["categorical_fields"]) - 1 + sum(
            name not in self.source_profile_sequences for name in source_schema["sequence_fields"])
        source_width = profile_dim + extra_count * dimension + len(source_schema["numeric_fields"])
        self.source_head = nn.Sequential(nn.Linear(source_width, 128), nn.ReLU(), nn.Dropout(.1),
                                         nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
        prior = min(max(source_schema["positive_rate"], 1e-6), 1 - 1e-6)
        nn.init.constant_(self.source_head[-1].bias, math.log(prior / (1 - prior)))

    def _ad_profile(self, lookup, sequences, sequence_indices):
        zero = torch.zeros_like(lookup["user_id"])
        return self.profile_tower(torch.cat([lookup["user_id"], lookup.get("feed_long_interest", zero),
                                             lookup.get("u_newsCatInterestsST", zero), lookup.get("feed_click_ca2", zero)], dim=-1))

    def forward(self, categories, sequences, numeric):
        fields, _, sequence_indices = self.encoder(categories, sequences)
        fields = bound_field_vectors(fields, self.field_groups, self.group_norm_caps)
        lookup = dict(zip(self.encoder.categorical_fields + self.encoder.sequence_fields, fields.unbind(dim=1)))
        profile = self._ad_profile(lookup, sequences, sequence_indices)
        dense = numeric if self.numeric_embeddings is None else self.numeric_embeddings(numeric).flatten(1)
        features = torch.cat([fields.flatten(1), dense], dim=-1)
        hidden = self.backbone(features[:, None].expand(-1, self.k, -1))
        expanded = profile[:, None].expand(-1, self.k, -1)
        weight = torch.sigmoid(self.transfer_gate(torch.cat([hidden, expanded], dim=-1)))
        hidden = hidden + weight * self.transfer_projection(profile)[:, None]
        return self.head(hidden).squeeze(-1)

    def _source_lookup(self, group, indices):
        if group in self.encoder.embeddings:
            table = self.encoder.embeddings[group]
            indices = self.encoder._indices(indices, table)
        else:
            table = self.source_embeddings[group]
            if self.training and table.num_embeddings > 100:
                replace = (indices >= 2) & (torch.rand(indices.shape, device=indices.device) < .05)
                indices = torch.where(replace, torch.ones_like(indices), indices)
        return table(indices)

    def _source_profile(self, vectors, history_vectors, sequences):
        user = bound_field_vectors(vectors["u_userId"][:, None], ["user_id"], self.group_norm_caps)[:, 0]
        return self.profile_tower(torch.cat([user] + [vectors[name] for name in self.source_profile_sequences], dim=-1))

    def forward_source(self, categories, sequences, numeric):
        schema = self.source_schema
        vectors = {}
        history_vectors = {}
        for index, name in enumerate(schema["categorical_fields"]):
            vectors[name] = self._source_lookup(schema["group_for_field"][name], categories[:, index])
        for index, name in enumerate(schema["sequence_fields"]):
            original = sequences[:, index]
            history_vectors[name] = self._source_lookup(schema["group_for_field"][name], original)
            vectors[name] = masked_mean(history_vectors[name], original.ne(0))
        profile = self._source_profile(vectors, history_vectors, sequences)
        pieces = [profile]
        pieces += [vectors[name] for name in schema["categorical_fields"] if name != "u_userId"]
        pieces += [vectors[name] for name in schema["sequence_fields"] if name not in self.source_profile_sequences]
        return self.source_head(torch.cat(pieces + [numeric], dim=-1)).squeeze(-1)
