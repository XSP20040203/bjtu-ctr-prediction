"""Shared, time-safe preprocessing helpers and the specified CTR neural model."""
from __future__ import annotations

import math
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch import nn


def category_key(value):
    if value is None or (isinstance(value, (float, np.floating)) and math.isnan(value)):
        return "__MISSING__"
    text = str(value).strip()
    try:
        number = float(text)
        if math.isfinite(number) and number.is_integer():
            return str(int(number))
    except ValueError:
        pass
    return text


def make_vocab(values):
    return {value: index + 2 for index, value in enumerate(sorted({category_key(value) for value in values}))}


def encode_sequence(value, vocabulary, length=5):
    result = np.zeros(length, dtype=np.int32)
    if value is None or str(value).strip() in ["", "nan", "None"]:
        return result
    for index, token in enumerate(str(value).split("^")[:length]):
        result[index] = vocabulary.get(category_key(token), 1)
    return result


def source_history_features(feeds, requests, alpha=20., initial_prior=.1):
    """Return seven source features using source rows strictly before each request day."""
    feeds = feeds.copy()
    feeds["u_userId"] = feeds["u_userId"].astype("int64")
    feeds["e_et"] = feeds["e_et"].astype("int64")
    feeds["_clicked"] = feeds["label"].eq(1).astype("int8")
    result = np.zeros((len(requests), 7), dtype=np.float32)
    days = requests["pt_d"].to_numpy(dtype=np.int64)
    users = requests["user_id"].to_numpy(dtype=np.int64)
    for day in np.unique(days):
        positions = np.flatnonzero(days == day)
        source = feeds.loc[feeds["e_et"].lt(day)]
        prior = float(source["_clicked"].mean()) if len(source) else initial_prior
        if source.empty:
            result[positions, 2] = prior
            continue
        aggregate = source.groupby("u_userId").agg(
            impressions=("label", "size"), clicks=("_clicked", "sum"),
            progress=("pro", "mean"), topics=("i_cat", "nunique"),
            active_days=("e_et", "nunique"), last_day=("e_et", "max"))
        aggregate.index = aggregate.index.astype("int64")
        aligned = aggregate.reindex(users[positions])
        counts = aligned["impressions"].fillna(0).to_numpy()
        clicks = aligned["clicks"].fillna(0).to_numpy()
        result[positions, 0] = counts
        result[positions, 1] = clicks
        result[positions, 2] = (clicks + alpha * prior) / (counts + alpha)
        result[positions, 3:6] = aligned[["progress", "topics", "active_days"]].fillna(0).to_numpy()
        last_days = aligned["last_day"].fillna(day).astype("int64")
        last_dates = pd.to_datetime(last_days.astype(str), format="%Y%m%d")
        gaps = (pd.Timestamp(datetime.strptime(str(day), "%Y%m%d")) - last_dates).dt.days.to_numpy()
        result[positions, 6] = gaps
    return result


def masked_mean(values, mask):
    weights = mask.unsqueeze(-1).to(values.dtype)
    return (values * weights).sum(dim=-2) / weights.sum(dim=-2).clamp_min(1.)


class CTRModel(nn.Module):
    def __init__(self, categorical_fields, sequence_fields, group_for_field, vocabulary_sizes,
                 numeric_dim=20, embedding_dim=16, hidden_dims=(256, 128, 64), dropout=.1,
                 unknown_dropout=.01, initial_ctr=None):
        super().__init__()
        self.categorical_fields = list(categorical_fields)
        self.sequence_fields = list(sequence_fields)
        self.group_for_field = group_for_field
        self.unknown_dropout = unknown_dropout
        self.embeddings = nn.ModuleDict({name: nn.Embedding(size, embedding_dim, padding_idx=0)
                                        for name, size in sorted(vocabulary_sizes.items())})
        for table in self.embeddings.values():
            nn.init.normal_(table.weight, mean=0., std=.02)
            with torch.no_grad():
                table.weight[0].zero_()
        width = (len(self.categorical_fields) + len(self.sequence_fields)) * embedding_dim + numeric_dim
        layers = []
        for index, hidden in enumerate(hidden_dims):
            layers.extend([nn.Linear(width, hidden), nn.ReLU()])
            if index < len(hidden_dims) - 1 and dropout:
                layers.append(nn.Dropout(dropout))
            width = hidden
        layers.append(nn.Linear(width, 1))
        self.mlp = nn.Sequential(*layers)
        if initial_ctr is not None:
            probability = min(max(initial_ctr, 1e-6), 1 - 1e-6)
            nn.init.constant_(self.mlp[-1].bias, math.log(probability / (1 - probability)))

    def _drop_unknown(self, indices, table):
        if self.training and self.unknown_dropout and table.num_embeddings > 100:
            replace = (indices >= 2) & (torch.rand(indices.shape, device=indices.device) < self.unknown_dropout)
            return torch.where(replace, torch.ones_like(indices), indices)
        return indices

    def forward(self, categories, sequences, numeric):
        pieces = []
        for index, name in enumerate(self.categorical_fields):
            table = self.embeddings[self.group_for_field[name]]
            pieces.append(table(self._drop_unknown(categories[:, index], table)))
        for index, name in enumerate(self.sequence_fields):
            table = self.embeddings[self.group_for_field[name]]
            original = sequences[:, index]
            indices = self._drop_unknown(original, table)
            pieces.append(masked_mean(table(indices), original.ne(0)))
        pieces.append(numeric)
        return self.mlp(torch.cat(pieces, dim=-1)).squeeze(-1)
