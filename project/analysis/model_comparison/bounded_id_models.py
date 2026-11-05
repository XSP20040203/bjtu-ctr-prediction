"""Limit effective ID-vector magnitudes while retaining their learned directions."""
import math

import torch

from id_regularized_models import IDRegularizedDCN


def bound_field_vectors(fields, field_groups, group_norm_caps):
    if not group_norm_caps:
        return fields
    vectors = []
    for index, group in enumerate(field_groups):
        vector = fields[:, index]
        if group in group_norm_caps:
            norm = vector.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            vector = vector * (group_norm_caps[group] / norm).clamp_max(1.)
        vectors.append(vector)
    return torch.stack(vectors, dim=1)


class BoundedIDDCN(IDRegularizedDCN):
    def __init__(self, *args, group_norm_caps=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.group_norm_caps = dict(group_norm_caps or {})
        for group, limit in self.group_norm_caps.items():
            if group not in self.encoder.embeddings or not math.isfinite(limit) or limit <= 0:
                raise ValueError(f"Invalid field norm bound: {group}={limit}")
        self.field_groups = [self.encoder.group_for_field[name]
                             for name in self.encoder.categorical_fields + self.encoder.sequence_fields]

    def forward(self, categories, sequences, numeric):
        fields, _, _ = self.encoder(categories, sequences)
        fields = bound_field_vectors(fields, self.field_groups, self.group_norm_caps)
        original = torch.cat([fields.flatten(start_dim=1), numeric], dim=-1)
        crossed = original
        for layer in self.cross:
            crossed = layer(original, crossed)
        return self.output(torch.cat([crossed, self.deep(original)], dim=-1)).squeeze(-1)
