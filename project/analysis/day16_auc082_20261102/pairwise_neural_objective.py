"""Class-conditional sampling and a smooth pairwise neural ranking objective."""
import numpy as np
import torch
from torch.nn import functional as F


def class_pools(labels, rows, dates, cutoff=20191116):
    rows = np.asarray(rows, dtype="int64")
    labels, dates = np.asarray(labels), np.asarray(dates)
    if not len(rows) or len(np.unique(rows)) != len(rows):
        raise ValueError("Training rows must be nonempty and unique")
    if np.any(dates[rows] >= cutoff) or not np.isin(labels[rows], [0, 1]).all():
        raise ValueError("Only observed strictly earlier advertisement labels may be sampled")
    positive, negative = rows[labels[rows] == 1], rows[labels[rows] == 0]
    if not len(positive) or not len(negative):
        raise ValueError("Both classes are required")
    return positive, negative


def sample_rows(positive, negative, rng, positive_count=64, negative_count=960):
    if min(positive_count, negative_count) < 1:
        raise ValueError("Both sample counts must be positive")
    return np.concatenate([rng.choice(positive, positive_count, replace=True),
                           rng.choice(negative, negative_count, replace=True)])


def ranking_loss(logits, positive_count, prior, temperature=1., bce_weight=.25,
                 positive_weight=5.):
    """Uniform class-conditional pair risk plus prior-corrected weighted BCE.

    Every positive is compared with every negative in the sampled batch. BCE
    weights restore the population class prior despite positive oversampling.
    """
    if (logits.ndim != 1 or not 0 < positive_count < len(logits)
            or not 0 < prior < 1 or temperature <= 0 or bce_weight < 0
            or positive_weight <= 0):
        raise ValueError("Invalid ranking objective arguments")
    positive, negative = logits[:positive_count], logits[positive_count:]
    pair = F.softplus((negative[None, :] - positive[:, None]) / temperature).mean()
    normalizer = 1. + (positive_weight - 1.) * prior
    bce = (positive_weight * prior * F.softplus(-positive).mean()
           + (1. - prior) * F.softplus(negative).mean()) / normalizer
    return pair + bce_weight * bce, pair, bce
