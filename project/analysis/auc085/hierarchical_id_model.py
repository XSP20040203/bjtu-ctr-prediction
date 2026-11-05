"""Official TabM with shared metadata priors and exposure-supported ID residuals.

The three current user/task/material fields are adapted. Existing list fields
retain their feature contract. Counts are input-only support, never click counts.
This is a custom composition, not an exact published model reproduction.
"""
import math

import numpy as np
import torch
from torch import nn

from modern_architectures import TabMFieldModel
from bounded_id_models import bound_field_vectors

SUPPORTED_FIELDS = ("user_id", "task_id", "adv_id")
DEFAULT_PRIORS = {
    "user_id": ["age", "sex", "residence", "city_rank", "device_name", "u_newsCatInterestsST",
                "feed_long_interest", "feed_clicked_topics", "feed_click_ca2"],
    "task_id": ["adv_prim_id", "spread_app_id", "app_second_class", "hispace_app_tags", "creat_type_cd", "inter_type_cd"],
    "adv_id": ["adv_prim_id", "spread_app_id", "app_second_class", "hispace_app_tags", "creat_type_cd", "inter_type_cd"],
}


def fit_exposure_support(categories, train_indices, manifest):
    """Count only current-field occurrences from supplied training X rows."""
    rows = np.asarray(train_indices)
    if rows.ndim != 1 or not np.issubdtype(rows.dtype, np.integer):
        raise ValueError("Training indices must be an integer vector")
    result = {}
    for field in SUPPORTED_FIELDS:
        if field not in manifest["categorical_fields"]:
            continue
        column = manifest["categorical_fields"].index(field)
        size = manifest["vocabulary_sizes"][manifest["group_for_field"][field]]
        result[field] = np.bincount(categories[rows, column], minlength=size).astype(np.int64)
        if len(result[field]) != size:
            raise ValueError(f"Category outside fitted vocabulary: {field}")
        result[field][:2] = 0
    return result


class HierarchicalIDTabM(TabMFieldModel):
    def __init__(self, *args, hierarchy_mode="shrink", support_temperature=200., prior_width=32,
                 hierarchy_priors=None, clip_numeric=True, **kwargs):
        super().__init__(*args, **kwargs)
        if hierarchy_mode not in ["raw", "metadata", "shrink", "shared"]:
            raise ValueError(f"Unknown hierarchy mode {hierarchy_mode}")
        if not math.isfinite(support_temperature) or support_temperature <= 0:
            raise ValueError("Initial support temperature must be positive")
        self.hierarchy_mode = hierarchy_mode
        names = self.encoder.categorical_fields + self.encoder.sequence_fields
        definitions = DEFAULT_PRIORS if hierarchy_priors is None else hierarchy_priors
        self.hierarchy_priors = {field: [name for name in definitions[field] if name in names]
                                 for field in SUPPORTED_FIELDS if field in self.encoder.categorical_fields}
        dimension = next(iter(self.encoder.embeddings.values())).embedding_dim
        self.prior_encoders = nn.ModuleDict()
        self.log_support_temperature = nn.ParameterDict()
        for field, priors in self.hierarchy_priors.items():
            if not priors or field in priors or set(priors).intersection(SUPPORTED_FIELDS):
                raise ValueError(f"Metadata prior must contain available non-ID fields: {field}")
            self.prior_encoders[field] = nn.Sequential(nn.Linear(len(priors) * dimension, prior_width),
                                                       nn.ReLU(), nn.Linear(prior_width, dimension))
            # The additive-metadata ablation starts exactly at the raw control.
            nn.init.zeros_(self.prior_encoders[field][-1].weight)
            nn.init.zeros_(self.prior_encoders[field][-1].bias)
            self.log_support_temperature[field] = nn.Parameter(torch.tensor(math.log(support_temperature)))
            table = self.encoder.embeddings[self.encoder.group_for_field[field]]
            self.register_buffer("support_counts_" + field, torch.zeros(table.num_embeddings, dtype=torch.float32))
        edges = kwargs.get("numeric_bin_edges")
        self.register_buffer("numeric_lower", torch.tensor([e[0] for e in edges], dtype=torch.float32) if clip_numeric and edges else None)
        self.register_buffer("numeric_upper", torch.tensor([e[-1] for e in edges], dtype=torch.float32) if clip_numeric and edges else None)

    def id_confidence(self, field, codes):
        count = getattr(self, "support_counts_" + field)[codes].float()
        temperature = self.log_support_temperature[field].clamp(math.log(1.), math.log(1e6)).exp()
        confidence = count / (count + temperature)
        return confidence * codes.ge(2)

    def encode_fields(self, categories, sequences):
        fields, category_indices, _ = self.encoder(categories, sequences)
        if self.hierarchy_mode == "raw":
            return bound_field_vectors(fields, self.field_groups, self.group_norm_caps)
        names = self.encoder.categorical_fields + self.encoder.sequence_fields
        lookup = dict(zip(names, fields.unbind(dim=1)))
        pieces = list(fields.unbind(dim=1))
        for field, prior_fields in self.hierarchy_priors.items():
            column = self.encoder.categorical_fields.index(field)
            prior = self.prior_encoders[field](torch.cat([lookup[name] for name in prior_fields], dim=-1))
            if self.hierarchy_mode == "shared":
                value = prior
            else:
                confidence = self.id_confidence(field, category_indices[column]) if self.hierarchy_mode == "shrink" else 1.
                value = prior + (confidence[:, None] * fields[:, column] if self.hierarchy_mode == "shrink" else fields[:, column])
            pieces[column] = value
        return bound_field_vectors(torch.stack(pieces, dim=1), self.field_groups, self.group_norm_caps)

    def forward(self, categories, sequences, numeric):
        fields = self.encode_fields(categories, sequences)
        if self.numeric_lower is not None:
            numeric = numeric.clamp(min=self.numeric_lower, max=self.numeric_upper)
        dense = numeric if self.numeric_embeddings is None else self.numeric_embeddings(numeric).flatten(1)
        features = torch.cat([fields.flatten(1), dense], dim=-1)
        hidden = self.backbone(features[:, None].expand(-1, self.k, -1))
        return self.head(hidden).squeeze(-1)
