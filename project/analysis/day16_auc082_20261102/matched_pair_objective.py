"""Sample observed positive-negative pairs within one user or user-day."""
import numpy as np
import torch
from torch.nn import functional as F
from pairwise_neural_objective import class_pools


def make_pool(labels, rows, dates, users, scope):
    positive, negative = class_pools(labels, rows, dates)
    if scope == "user":
        keys = np.asarray(users, dtype="int64")
    elif scope == "userday":
        _, keys = np.unique(np.column_stack([dates, users]), axis=0, return_inverse=True)
    else:
        raise ValueError("Unknown matched scope")
    neg_keys = keys[negative]
    order = np.argsort(neg_keys, kind="stable")
    sorted_keys, sorted_neg = neg_keys[order], negative[order]
    starts = np.searchsorted(sorted_keys, keys[positive], side="left")
    ends = np.searchsorted(sorted_keys, keys[positive], side="right")
    eligible = ends > starts
    if not np.any(eligible):
        raise ValueError("No observed positive has a negative in its group")
    return {"positive": positive[eligible], "start": starts[eligible], "end": ends[eligible],
            "negative": sorted_neg, "keys": keys, "all_positive": positive, "all_negative": negative}


def sample(pool, rng, positive_count=64, negatives_per_positive=7, uniform_negative_count=512):
    chosen = rng.integers(len(pool["positive"]), size=positive_count)
    positive = pool["positive"][chosen]
    widths = pool["end"][chosen]-pool["start"][chosen]
    offsets = np.floor(rng.random((positive_count, negatives_per_positive))*widths[:, None]).astype("int64")
    negative = pool["negative"][(pool["start"][chosen, None]+offsets).ravel()]
    uniform = rng.choice(pool["all_negative"], uniform_negative_count, replace=True)
    return np.r_[positive, negative, uniform]


def objective(logits, positive_count, negatives_per_positive, prior, temperature=1.):
    stop = positive_count*(1+negatives_per_positive)
    positive = logits[:positive_count]
    matched = logits[positive_count:stop].reshape(positive_count, negatives_per_positive)
    uniform = logits[stop:]
    if not len(uniform):
        raise ValueError("Uniform negative BCE reference is required")
    pair = F.softplus((matched-positive[:, None])/temperature).mean()
    # Positive eligibility depends on group exposure, so this is an explicit
    # regularizer rather than an unbiased estimate of population logloss.
    bce = (5.*prior*F.softplus(-positive).mean()+(1.-prior)*F.softplus(uniform).mean())/(1.+4.*prior)
    return pair+.25*bce, pair, bce
