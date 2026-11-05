"""Train-only tabular normalization and row-level factorization interactions."""
import numpy as np
import torch
from torch import nn


def fit_schema(frame, categories):
    numeric = [name for name in frame if name not in categories]
    values = frame[numeric].to_numpy(dtype='float64')
    if not np.isfinite(values).all():
        raise ValueError('Finite prior training values required')
    return dict(columns=list(frame), categories=list(categories), numeric=numeric,
                mean=values.mean(axis=0).tolist(), scale=np.maximum(values.std(axis=0), 1e-4).tolist(),
                vocabularies={name: np.sort(frame[name].unique()).astype('int64').tolist() for name in categories})


def transform(frame, schema, independent=False):
    if list(frame) != schema['columns']:
        raise ValueError('Fitted feature order required')
    numeric = ((frame[schema['numeric']].to_numpy(dtype='float64') - np.asarray(schema['mean'])) /
               np.asarray(schema['scale']))
    numeric = np.clip(numeric, -10., 10.).astype('float32')
    categories = []
    for name in schema['categories']:
        vocab = np.asarray(schema['vocabularies'][name], dtype='int64')
        values = frame[name].to_numpy(dtype='int64')
        if independent:
            mapping = {int(value): i + 1 for i, value in enumerate(vocab)}
            encoded = np.fromiter((mapping.get(int(value), 0) for value in values), dtype='int64', count=len(values))
        else:
            position = np.searchsorted(vocab, values)
            known = (position < len(vocab)) & (vocab[np.minimum(position, len(vocab) - 1)] == values)
            encoded = np.where(known, position + 1, 0)
        categories.append(encoded)
    encoded = np.column_stack(categories).astype('int64')
    if not np.isfinite(numeric).all():
        raise ValueError('Nonfinite normalized features')
    return numeric, encoded


class DeepFM(nn.Module):
    def __init__(self, numeric, cardinalities, embedding_dim, dropout=.2):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(n + 1, embedding_dim, padding_idx=0) for n in cardinalities])
        self.linear_embeddings = nn.ModuleList([nn.Embedding(n + 1, 1, padding_idx=0) for n in cardinalities])
        self.numeric_linear = nn.Linear(numeric, 1)
        width = numeric + len(cardinalities) * embedding_dim
        self.network = nn.Sequential(nn.Linear(width, 192), nn.LayerNorm(192), nn.SiLU(), nn.Dropout(dropout),
                                     nn.Linear(192, 96), nn.LayerNorm(96), nn.SiLU(), nn.Dropout(dropout),
                                     nn.Linear(96, 1))
        for embedding in self.embeddings:
            nn.init.normal_(embedding.weight, 0., .02)
            with torch.no_grad():
                embedding.weight[0].zero_()
        for embedding in self.linear_embeddings:
            nn.init.zeros_(embedding.weight)

    def forward(self, numeric, categories):
        embeddings = torch.stack([layer(categories[:, i]) for i, layer in enumerate(self.embeddings)], dim=1)
        first = self.numeric_linear(numeric).squeeze(-1)
        first = first + torch.stack([layer(categories[:, i]).squeeze(-1)
                                    for i, layer in enumerate(self.linear_embeddings)], dim=1).sum(dim=1)
        interaction = .5 * ((embeddings.sum(dim=1) ** 2) - (embeddings ** 2).sum(dim=1)).sum(dim=1)
        hidden = torch.cat([numeric, embeddings.flatten(1)], dim=1)
        return first + interaction + self.network(hidden).squeeze(-1)


def prior_rows(dates, labels, cutoff):
    dates, labels = np.asarray(dates), np.asarray(labels)
    if dates.shape != labels.shape or np.any(labels[dates >= cutoff] != -1):
        raise ValueError('Current and future targets must be hidden')
    rows = np.flatnonzero(dates < cutoff)
    if not len(rows) or not np.isin(labels[rows], [0, 1]).all():
        raise ValueError('Observed binary prior targets required')
    return rows
