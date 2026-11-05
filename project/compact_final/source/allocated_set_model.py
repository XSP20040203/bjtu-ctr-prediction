"""Allocate a predicted user-day count across its candidates, without labels.

The unclipped row intensities sum to the parent's predicted count in each set.
For Bernoulli supervision, intensity is mapped to event probability 1-exp(-x).
Only this final conversion clips intensity to [1e-8, 80] for finite logits.
"""
from __future__ import annotations

import math

import torch

try:
    from .set_context_model import SetContextCTR
except ImportError:
    from set_context_model import SetContextCTR


def allocate_intensity(row_scores, log_lambda, group_index, temperature=1.0):
    """Return lambda_g * softmax(score / temperature within g), shape [B].

    Groups must be nonempty and contiguous 0..G-1, in any row order. Accumulation
    uses float32 (float64 for double inputs), independent of autocast. Log counts
    are expected to come from a bounded count head: lambda_g <= group size.
    No clipping is applied here, so the group mass is conserved up to rounding.
    """
    if row_scores.ndim != 1 or log_lambda.ndim != 1 or group_index.ndim != 1:
        raise ValueError("scores, log_lambda and group_index must be one-dimensional")
    rows = row_scores.numel()
    if rows == 0 or log_lambda.numel() == 0:
        raise ValueError("Empty group input: at least one row and group are required")
    if group_index.numel() != rows or group_index.dtype not in (torch.int32, torch.int64):
        raise ValueError("group_index must contain one integer group ID per row")
    if any(t.device != row_scores.device for t in (log_lambda, group_index)):
        raise ValueError("All allocation inputs must be on the same device")
    if not row_scores.is_floating_point() or not log_lambda.is_floating_point():
        raise ValueError("row_scores and log_lambda must be floating point")
    groups = group_index.long()
    group_count = log_lambda.numel()
    if group_count > rows or int(groups.min()) != 0 or int(groups.max()) != group_count - 1:
        raise ValueError("group_index must be contiguous from 0 to len(log_lambda)-1")
    counts = torch.bincount(groups, minlength=group_count)
    if bool((counts == 0).any()):
        raise ValueError("group_index cannot contain empty groups")
    dtype = torch.float64 if torch.float64 in (row_scores.dtype, log_lambda.dtype) else torch.float32
    scale = torch.as_tensor(temperature, device=row_scores.device, dtype=dtype)
    if scale.ndim != 0 or not bool(torch.isfinite(scale)) or not bool(scale > 0):
        raise ValueError("temperature must be a finite positive scalar")
    scores = row_scores.to(dtype) / scale
    # A shifted segment logsumexp avoids overflow and cancellation when all
    # scores in a group have a large common offset. No loop over groups.
    maxima = scores.new_full((group_count,), -torch.inf)
    maxima.scatter_reduce_(0, groups, scores.detach(), reduce="amax", include_self=True)
    shifted = scores - maxima[groups]
    normalizer = scores.new_zeros(group_count).index_add_(0, groups, shifted.exp())
    log_weights = shifted - normalizer.log()[groups]
    return (log_lambda.to(dtype)[groups] + log_weights).exp()


class AllocatedSetContextCTR(SetContextCTR):
    """SetContextCTR whose row BCE also supervises its predicted group count.

    Constructor and forward match the existing runner. ``temperature`` controls
    within-set allocation and is checkpointed as a buffer, adding no parameters.
    Targets are supplied only to the caller's row-BCE and count-Poisson losses.
    """

    def __init__(self, *args, temperature=1.0, **kwargs):
        value = float(temperature)
        if not math.isfinite(value) or value <= 0:
            raise ValueError("temperature must be finite and positive")
        super().__init__(*args, **kwargs)
        self.register_buffer("allocation_temperature", torch.tensor(value))

    def forward(self, categories, sequences, numeric, group_index):
        row_scores, log_lambda = super().forward(categories, sequences, numeric, group_index)
        intensity = allocate_intensity(row_scores, log_lambda, group_index,
                                       temperature=self.allocation_temperature)
        intensity = intensity.clamp(min=1e-8, max=80.)
        # log(expm1(x)) = x + log(1-exp(-x)); expm1 keeps tiny x accurate.
        row_logits = intensity + torch.log(-torch.expm1(-intensity))
        return row_logits, log_lambda
