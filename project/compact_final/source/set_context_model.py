"""Label-free, linear-cost context over complete user-day candidate sets.

Callers must provide every candidate in each user-day together. Group IDs only
identify sets; their numeric values and ordering are not model features. The
parameter budget includes shared vocabulary tables (default: fewer than 1M).
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

try:
    from .interaction_models import CappedInteractionEncoder
except ImportError:  # Direct execution/import by the training runner.
    from interaction_models import CappedInteractionEncoder


class SetContextCTR(nn.Module):
    """Row logits and expected positive counts from complete, unordered sets.

    Row tokens use flattened shared field embeddings plus a 32-wide numeric
    projection, followed by a 128 -> 64 MLP. Four learned slots pool each set;
    each candidate attends to those slots. All reductions are vectorized and
    work with interleaved group IDs. No labels or target aggregates are inputs.
    """

    def __init__(self, categorical_fields, sequence_fields, group_for_field,
                 vocabulary_sizes, numeric_dim=500, initial_ctr=None, *,
                 embedding_dim=None, unknown_dropout=.01, group_dropout_rates=None,
                 group_norm_caps=None, history_queries=None, include_max=True,
                 use_set_context=True, max_parameters=999_999):
        super().__init__()
        if numeric_dim <= 0:
            raise ValueError("numeric_dim must be positive")
        probability = .5 if initial_ctr is None else float(initial_ctr)
        if not math.isfinite(probability) or not 0 < probability < 1:
            raise ValueError("initial_ctr must be finite and strictly between 0 and 1")
        categorical_fields, sequence_fields = list(categorical_fields), list(sequence_fields)
        field_count = len(categorical_fields) + len(sequence_fields)
        if not field_count:
            raise ValueError("At least one categorical or sequence field is required")
        self.numeric_dim = numeric_dim
        self.include_max = include_max
        self.use_set_context = bool(use_set_context)
        self.register_buffer("initial_logit", torch.tensor(math.log(probability / (1 - probability))))

        self.numeric_projection = nn.Sequential(nn.Linear(numeric_dim, 32), nn.LayerNorm(32), nn.SiLU())
        self.slots = nn.Parameter(torch.empty(4, 32))
        nn.init.normal_(self.slots, std=.1)
        self.pool_key = nn.Linear(64, 32, bias=False)
        self.pool_value = nn.Linear(64, 64, bias=False)
        self.candidate_query = nn.Linear(64, 32, bias=False)
        self.slot_key = nn.Linear(64, 32, bias=False)
        self.row_head = nn.Linear(64, 1)
        self.context_head = nn.Sequential(nn.Linear(64 * 4, 64), nn.SiLU(), nn.Linear(64, 1))
        summary_dim = 64 * (3 if include_max else 2) + 1
        self.group_bias = nn.Sequential(nn.Linear(summary_dim, 32), nn.SiLU(), nn.Linear(32, 1))
        self.count_head = nn.Sequential(nn.Linear(summary_dim, 32), nn.SiLU(), nn.Linear(32, 1))

        # Choose the largest width up to eight that fits the TOTAL budget,
        # including vocabularies and the field-dependent first row layer.
        fixed = sum(p.numel() for p in self.parameters())
        vocab_rows = sum(vocabulary_sizes.values())
        row_fixed = 128 * 32 + 128 + 2 * 128 + 64 * 128 + 64 + 2 * 64
        history_count = len(history_queries or {})
        per_dimension = vocab_rows + 128 * field_count + 32 * history_count
        available = (max_parameters - fixed - row_fixed) // max(1, per_dimension)
        width = min(8, available) if embedding_dim is None else int(embedding_dim)
        if width < 1 or width > available:
            raise ValueError("Vocabulary/embedding width exceeds the total parameter budget")
        self.embedding_dim = width
        self.encoder = CappedInteractionEncoder(
            categorical_fields, sequence_fields, group_for_field, vocabulary_sizes,
            embedding_dim=width, unknown_dropout=unknown_dropout,
            group_dropout_rates=group_dropout_rates, group_norm_caps=group_norm_caps,
            history_queries=history_queries)
        self.row_mlp = nn.Sequential(
            nn.Linear(field_count * width + 32, 128), nn.SiLU(), nn.LayerNorm(128),
            nn.Linear(128, 64), nn.SiLU(), nn.LayerNorm(64))
        nn.init.zeros_(self.row_head.bias)
        nn.init.normal_(self.context_head[-1].weight, std=.001)
        nn.init.zeros_(self.context_head[-1].bias)
        nn.init.zeros_(self.group_bias[-1].weight)
        nn.init.zeros_(self.group_bias[-1].bias)
        nn.init.zeros_(self.count_head[-1].weight)
        nn.init.zeros_(self.count_head[-1].bias)
        self.parameter_count = sum(p.numel() for p in self.parameters())
        if self.parameter_count > max_parameters:
            raise ValueError("Model exceeds the total parameter budget")

    def _validate_groups(self, categories, sequences, numeric, group_index):
        rows = numeric.shape[0]
        if rows == 0:
            raise ValueError("Empty group input: at least one row is required")
        if categories.shape != (rows, len(self.encoder.categorical_fields)):
            raise ValueError("categories must have shape [B, categorical_fields]")
        if (sequences.ndim != 3 or sequences.shape[:2] !=
                (rows, len(self.encoder.sequence_fields))):
            raise ValueError("sequences must have shape [B, sequence_fields, history_length]")
        if numeric.ndim != 2 or numeric.shape[1] != self.numeric_dim:
            raise ValueError("numeric must have shape [B, numeric_dim]")
        if group_index.ndim != 1 or group_index.shape[0] != rows:
            raise ValueError("group_index must have shape [B]")
        if group_index.dtype not in (torch.int32, torch.int64):
            raise ValueError("group_index must contain integer group IDs")
        if any(t.device != numeric.device for t in (categories, sequences, group_index)):
            raise ValueError("All inputs must be on the same device")
        groups = group_index.long()
        low, high = int(groups.min()), int(groups.max())
        if low != 0 or high >= rows:
            raise ValueError("group_index must be contiguous from 0 to G-1")
        counts = torch.bincount(groups, minlength=high + 1)
        if bool((counts == 0).any()):
            raise ValueError("group_index must be contiguous from 0 to G-1 without empty groups")
        return groups, counts

    def forward(self, categories, sequences, numeric, group_index):
        groups, counts = self._validate_groups(categories, sequences, numeric, group_index)
        fields, _, _ = self.encoder(categories, sequences)
        projected = self.numeric_projection(numeric)
        tokens = self.row_mlp(torch.cat((fields.flatten(1), projected), dim=-1))
        group_count = counts.numel()
        # Accumulate in float32 for stable large sets and mixed precision.
        work = tokens.float()
        mean = work.new_zeros((group_count, 64)).index_add_(0, groups, work)
        mean = mean / counts.float().unsqueeze(1)
        scores = (self.pool_key(tokens).float() @ self.slots.float().T) / math.sqrt(32)
        indices = groups[:, None].expand(-1, 4)
        maxima = scores.new_full((group_count, 4), -torch.inf)
        maxima.scatter_reduce_(0, indices, scores.detach(), reduce="amax", include_self=True)
        weights = (scores - maxima[groups]).exp()
        denominator = weights.new_zeros((group_count, 4)).index_add_(0, groups, weights)
        weights = weights / denominator[groups].clamp_min(torch.finfo(weights.dtype).tiny)
        values = self.pool_value(tokens).float()
        pooled_slots = values.new_zeros((group_count, 4, 64)).index_add_(
            0, groups, weights[:, :, None] * values[:, None, :])
        slots = pooled_slots.to(tokens.dtype)
        candidate_scores = (self.candidate_query(tokens)[:, None, :] *
                            self.slot_key(slots)[groups]).sum(-1).float() / math.sqrt(32)
        candidate_weights = candidate_scores.softmax(-1)
        context = (candidate_weights[:, :, None] * pooled_slots[groups]).sum(1).to(tokens.dtype)
        row_mean = mean[groups].to(tokens.dtype)
        residual = self.context_head(torch.cat((tokens, row_mean, context, tokens * context), dim=-1)).squeeze(-1)
        log_size = counts.float().log()
        summary_parts = [mean, pooled_slots.mean(1)]
        if self.include_max:
            maximum = work.new_full((group_count, 64), -torch.inf)
            maximum.scatter_reduce_(0, groups[:, None].expand(-1, 64), work,
                                    reduce="amax", include_self=True)
            summary_parts.append(maximum)
        summary_parts.append(log_size[:, None])
        summary = torch.cat(summary_parts, dim=-1).to(tokens.dtype)
        row_logits = self.initial_logit + self.row_head(tokens).squeeze(-1)
        if self.use_set_context:
            row_logits = row_logits + residual + self.group_bias(summary).squeeze(-1)[groups]
        count_logit = self.initial_logit.float() + .1 * self.count_head(summary).squeeze(-1).float()
        # log(n * sigmoid(x)), stable even when sigmoid(x) would underflow.
        log_lambda = log_size + F.logsigmoid(count_logit)
        return row_logits, log_lambda
