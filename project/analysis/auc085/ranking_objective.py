"""Optional within-batch ranking loss, separate from the calibration BCE."""
import torch
from torch.nn import functional as F


def member_ranking_loss(logits, labels, max_negatives=128):
    if logits.ndim != 2 or labels.ndim != 1 or len(logits) != len(labels) or max_negatives < 1:
        raise ValueError("Ranking requires member logits, matching binary labels and a positive negative cap")
    positive = logits[labels.eq(1)].float()
    negative = logits[labels.eq(0)][:max_negatives].float()
    if not len(positive) or not len(negative):
        return logits.float().sum() * 0.
    # The training rows are already shuffled independently each epoch. Taking
    # the first negatives adds no RNG effects to dropout in the BCE control.
    return F.softplus(negative[None] - positive[:, None]).mean()
