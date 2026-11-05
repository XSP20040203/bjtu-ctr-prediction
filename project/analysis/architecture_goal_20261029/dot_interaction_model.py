"""DLRM-style CTR: retain every distinct vector dot product for the top MLP.

No labels, data reads, or cross-row statistics are used. The numeric bottom MLP
creates one dense field; raw numeric columns never enter the interaction MLP.
"""
from __future__ import annotations

import math

import torch
from torch import nn

try:
    from .interaction_models import CappedInteractionEncoder, _DEFAULT_QUERIES
except ImportError:
    from interaction_models import CappedInteractionEncoder, _DEFAULT_QUERIES


class PairwiseDotInteraction(nn.Module):
    """Strict lower triangle, in row-major order, divided by sqrt(vector width)."""

    def __init__(self, field_count):
        super().__init__()
        if field_count < 2:
            raise ValueError("At least two fields are required")
        self.field_count = field_count
        self.output_dim = field_count * (field_count - 1) // 2
        self.register_buffer("pair_indices", torch.tril_indices(field_count, field_count, offset=-1))

    def forward(self, fields):
        if fields.ndim != 3 or fields.shape[1] != self.field_count or fields.shape[2] == 0:
            raise ValueError("Expected [batch, field_count, positive_embedding_dim]")
        gram = torch.bmm(fields, fields.transpose(1, 2))
        return gram[:, self.pair_indices[0], self.pair_indices[1]] / math.sqrt(fields.shape[-1])


class DotInteractionCTR(nn.Module):
    """Shared capped embeddings, target attention, numeric token, and explicit dots.

    ``numeric_dim=None`` falls back to 166; zero numeric columns are supported.
    Field gains are bounded by 2*sigmoid. Per-vector RMS normalization uses no
    batch statistics or affine bias. Sensitive group caps are applied AGAIN
    after normalization and field gains, so neither operation bypasses them.
    The numeric linear branch is a scalar residual to the final logit.
    """

    def __init__(self, categorical_fields, sequence_fields, group_for_field, vocabulary_sizes,
                 numeric_dim=166, initial_ctr=None, embedding_dim=16,
                 hidden_dims=(256, 128, 64), dropout=.15, unknown_dropout=.01,
                 group_dropout_rates=None, group_norm_caps=None, target_attention=True,
                 history_queries=None, attention_dim=16, bottom_hidden_dim=128,
                 bottom_layer_norm=True, numeric_clip=8., rms_epsilon=1e-8):
        super().__init__()
        numeric_dim = 166 if numeric_dim is None else numeric_dim
        if (not isinstance(numeric_dim, int) or numeric_dim < 0 or embedding_dim <= 0
                or bottom_hidden_dim <= 0 or not hidden_dims or any(w <= 0 for w in hidden_dims)
                or not 0 <= dropout < 1 or not math.isfinite(numeric_clip) or numeric_clip <= 0
                or not math.isfinite(rms_epsilon) or rms_epsilon <= 0):
            raise ValueError("Invalid dimensions, dropout, numeric clip, or RMS epsilon")
        if initial_ctr is not None and (not math.isfinite(initial_ctr) or not 0 < initial_ctr < 1):
            raise ValueError("initial_ctr must be finite and strictly between zero and one")
        if history_queries is None:
            history_queries = {s: c for s, c in _DEFAULT_QUERIES.items()
                               if s in sequence_fields and c in categorical_fields}
        self.encoder = CappedInteractionEncoder(
            categorical_fields, sequence_fields, group_for_field, vocabulary_sizes,
            embedding_dim=embedding_dim, unknown_dropout=unknown_dropout,
            group_dropout_rates=group_dropout_rates, group_norm_caps=group_norm_caps,
            history_queries=history_queries if target_attention else {}, attention_dim=attention_dim)
        self.numeric_dim = numeric_dim
        self.embedding_dim = embedding_dim
        self.numeric_clip = float(numeric_clip)
        self.rms_epsilon = float(rms_epsilon)
        self.field_names = self.encoder.categorical_fields + self.encoder.sequence_fields
        self.field_count = len(self.field_names) + 1
        self.field_scale_logits = nn.Parameter(torch.zeros(self.field_count))
        # Zero marks uncapped fields; infinity in cap/norm creates NaN backward.
        caps = [self.encoder.group_norm_caps.get(group_for_field[name], 0.)
                for name in self.field_names]
        self.register_buffer("field_norm_caps", torch.tensor(caps).reshape(1, -1, 1))
        if numeric_dim:
            layers = [nn.Linear(numeric_dim, bottom_hidden_dim), nn.ReLU(),
                      nn.Linear(bottom_hidden_dim, embedding_dim)]
            if bottom_layer_norm:
                layers.append(nn.LayerNorm(embedding_dim))
            self.bottom = nn.Sequential(*layers)
            self.numeric_linear = nn.Linear(numeric_dim, 1, bias=False)
            nn.init.normal_(self.numeric_linear.weight, std=.001)
        else:
            self.bottom = None
            self.numeric_linear = None
        self.interaction = PairwiseDotInteraction(self.field_count)
        width = self.interaction.output_dim + embedding_dim
        layers = []
        for hidden in hidden_dims:
            layers.extend([nn.Linear(width, hidden), nn.ReLU(), nn.Dropout(dropout)])
            width = hidden
        self.top = nn.Sequential(*layers)
        self.output = nn.Linear(width, 1, bias=False)
        nn.init.normal_(self.output.weight, std=.001)
        prior = .5 if initial_ctr is None else initial_ctr
        self.bias = nn.Parameter(torch.tensor(math.log(prior / (1 - prior)), dtype=torch.float32))

    def _numeric_features(self, numeric):
        if numeric.ndim != 2 or numeric.shape[-1] != self.numeric_dim:
            raise ValueError(f"Expected numeric shape [batch, {self.numeric_dim}]")
        return torch.nan_to_num(numeric, nan=0., posinf=self.numeric_clip,
                                neginf=-self.numeric_clip).clamp(-self.numeric_clip, self.numeric_clip)

    def _scaled_fields(self, fields):
        # Float32 accumulation keeps epsilon representable under mixed precision.
        work = fields.float() if fields.dtype in (torch.float16, torch.bfloat16) else fields
        normalized = work * torch.rsqrt(work.square().mean(-1, keepdim=True) + self.rms_epsilon)
        scaled = normalized * (2 * torch.sigmoid(self.field_scale_logits[:-1])).reshape(1, -1, 1)
        norm = scaled.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        cap_scale = (self.field_norm_caps / norm).clamp_max(1.)
        scaled = scaled * torch.where(self.field_norm_caps > 0, cap_scale, torch.ones_like(cap_scale))
        return scaled.to(fields.dtype)

    def field_tokens(self, categories, sequences, numeric):
        """Return [B, categorical+history+1, D] tokens and sanitized numerics."""
        fields, _, _ = self.encoder(categories, sequences)
        clipped = self._numeric_features(numeric)
        dense = (self.bottom(clipped) if self.bottom is not None
                 else fields.new_zeros(fields.shape[0], self.embedding_dim))
        dense = dense * (2 * torch.sigmoid(self.field_scale_logits[-1]))
        tokens = torch.cat([self._scaled_fields(fields), dense.unsqueeze(1)], dim=1)
        return tokens, clipped

    def forward(self, categories, sequences, numeric):
        tokens, clipped = self.field_tokens(categories, sequences, numeric)
        top_input = torch.cat([tokens[:, -1], self.interaction(tokens)], dim=-1)
        logits = self.bias + self.output(self.top(top_input)).squeeze(-1)
        if self.numeric_linear is not None:
            logits = logits + self.numeric_linear(clipped).squeeze(-1)
        return logits


DLRMCTR = DotInteractionCTR

__all__ = ["DotInteractionCTR", "DLRMCTR", "PairwiseDotInteraction"]
