"""Compact piecewise numeric alternative to GatedCrossCTR.

The caller fits ``fit_numeric_bin_edges(training_sample)`` using outer-training
rows only, then supplies those edges as ``numeric_bin_edges``. Neither model
construction nor forward fits bins, loads data, or accepts labels. Checkpoints
include fixed bin transforms, clipping bounds, and constant-field masks.
"""
from __future__ import annotations

import math
import warnings

import torch
from torch import nn
from rtdl_num_embeddings import PiecewiseLinearEmbeddings

from interaction_models import GatedCrossCTR


def _clip_numeric(values, clip):
    return torch.nan_to_num(values, nan=0., posinf=clip, neginf=-clip).clamp(-clip, clip)


def fit_numeric_bin_edges(training_sample, n_bins=8, numeric_clip=8.):
    """Return up to n_bins quantile intervals per column, without labels or IO.

    Input must already contain only the selected outer-training numeric rows in
    the same scaled column order used at inference. Sampling is the caller's
    responsibility. A constant column returns a single edge; the encoder masks
    its piecewise branch while the original-value linear branch remains valid.
    Nonfinite values follow the same replacement and clipping as forward.
    """
    if not isinstance(n_bins, int) or isinstance(n_bins, bool) or not 2 <= n_bins <= 8:
        raise ValueError("n_bins must be an integer between 2 and 8")
    if not math.isfinite(numeric_clip) or numeric_clip <= 0:
        raise ValueError("numeric_clip must be finite and positive")
    sample = torch.as_tensor(training_sample).detach().to(device="cpu", dtype=torch.float32)
    if sample.ndim != 2 or sample.shape[0] < 2 or sample.shape[1] < 1:
        raise ValueError("training_sample must have at least two rows and one column")
    sample = _clip_numeric(sample, float(numeric_clip))
    quantiles = torch.quantile(sample, torch.linspace(0., 1., n_bins + 1), dim=0)
    return [column.unique(sorted=True).clone() for column in quantiles.T]


class GatedPiecewiseNumeric(nn.Module):
    """Two learned coordinates per field, independent gains, then a compact mix."""

    def __init__(self, numeric_dim, numeric_bin_edges, projection_dim=64,
                 numeric_clip=8., layer_norm=False, embedding_std=.02):
        super().__init__()
        if not isinstance(numeric_dim, int) or numeric_dim <= 0:
            raise ValueError("numeric_dim must be a positive integer")
        if not isinstance(projection_dim, int) or projection_dim <= 0:
            raise ValueError("projection_dim must be a positive integer")
        if not math.isfinite(numeric_clip) or numeric_clip <= 0:
            raise ValueError("numeric_clip must be finite and positive")
        if not math.isfinite(embedding_std) or embedding_std <= 0:
            raise ValueError("embedding_std must be finite and positive")
        if numeric_bin_edges is None or len(numeric_bin_edges) != numeric_dim:
            raise ValueError("Supply one training-fitted bin-edge vector per numeric field")
        self.numeric_dim = numeric_dim
        self.projection_dim = projection_dim
        self.clip = float(numeric_clip)
        bins, lower, upper, constant = [], [], [], []
        for index, supplied in enumerate(numeric_bin_edges):
            edge = torch.as_tensor(supplied).detach().to(device="cpu", dtype=torch.float32).clone()
            if edge.ndim != 1 or not 1 <= edge.numel() <= 9:
                raise ValueError(f"Field {index}: expected 1 through 9 bin edges")
            if not torch.isfinite(edge).all() or (edge.abs() > self.clip).any():
                raise ValueError(f"Field {index}: edges must be finite and inside numeric_clip")
            if edge.numel() > 1:
                width = edge.diff()
                if (width <= 0).any() or not torch.isfinite(width.reciprocal()).all():
                    raise ValueError(f"Field {index}: edges must be strictly increasing with finite inverse widths")
            is_constant = edge.numel() == 1
            lower.append(edge[0])
            upper.append(edge[-1])
            constant.append(is_constant)
            # The library requires two edges. A constant field is masked below,
            # so its dummy interval has no effect on predictions or gradients.
            bins.append(torch.tensor([-1., 1.]) if is_constant else edge)
        self.register_buffer("lower_bounds", torch.stack(lower))
        self.register_buffer("upper_bounds", torch.stack(upper))
        self.register_buffer("constant_mask", torch.tensor(constant, dtype=torch.bool))
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r"The \d+-th feature has just two bin edges.*")
            self.embeddings = PiecewiseLinearEmbeddings(bins, 2, activation=False, version="A")
        nn.init.normal_(self.embeddings.linear.weight, std=embedding_std)
        nn.init.zeros_(self.embeddings.linear.bias)
        self.gate_logits = nn.Parameter(torch.zeros(numeric_dim))
        self.feature_scale = nn.Parameter(torch.ones(numeric_dim))
        self.projection = nn.Linear(numeric_dim * 2, projection_dim)
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)
        self.normalizer = nn.LayerNorm(projection_dim) if layer_norm else nn.Identity()
        self.activation = nn.SiLU()

    def clipped_values(self, numeric):
        if numeric.ndim != 2 or numeric.shape[1] != self.numeric_dim:
            raise ValueError(f"numeric must have shape (batch, {self.numeric_dim})")
        return _clip_numeric(numeric, self.clip)

    def embed_fields(self, numeric):
        clipped = self.clipped_values(numeric)
        # The installed rtdl version extrapolates its first/last bin unless the
        # inputs are explicitly clamped to the fitted per-field endpoints.
        bounded = torch.maximum(torch.minimum(clipped, self.upper_bounds), self.lower_bounds)
        embedded = self.embeddings(bounded)
        gains = (2. * self.gate_logits.sigmoid()) * self.feature_scale
        gains = gains * (~self.constant_mask).to(gains.dtype)
        return embedded * gains[None, :, None]

    def forward(self, numeric):
        fields = self.embed_fields(numeric)
        return self.activation(self.normalizer(self.projection(fields.flatten(1))))


class PiecewiseGatedCTR(GatedCrossCTR):
    """GatedCrossCTR with a 64-wide numeric input and an original-value residual.

    Defaults: 166 numeric fields, two embedding coordinates, eight quantile bins,
    two rank-32/three-expert cross layers and a (128, 64) deep branch. Embedding
    tables, PAD handling, ID caps and group dropout are inherited unchanged.
    The single inherited bias carries initial_ctr; the numeric residual has no
    bias and starts at zero. Parameter count also depends on vocabulary sizes.
    """

    def __init__(self, categorical_fields, sequence_fields, group_for_field, vocabulary_sizes,
                 numeric_dim=166, numeric_bin_edges=None, initial_ctr=None,
                 numeric_projection_dim=64, numeric_layer_norm=False,
                 numeric_embedding_std=.02, embedding_dim=8, hidden_dims=(128, 64),
                 dropout=.1, unknown_dropout=.01, group_dropout_rates=None,
                 group_norm_caps=None, cross_layers=2, cross_rank=32, cross_experts=3,
                 cross_layer_norm=False, numeric_hidden_dim=16, numeric_clip=8.,
                 se_hidden_dim=16):
        # Validate supplied edges before allocating the larger shared model.
        numeric_encoder = GatedPiecewiseNumeric(
            numeric_dim, numeric_bin_edges, numeric_projection_dim, numeric_clip,
            numeric_layer_norm, numeric_embedding_std)
        super().__init__(
            categorical_fields, sequence_fields, group_for_field, vocabulary_sizes,
            numeric_dim=numeric_projection_dim, initial_ctr=initial_ctr,
            embedding_dim=embedding_dim, hidden_dims=hidden_dims, dropout=dropout,
            unknown_dropout=unknown_dropout, group_dropout_rates=group_dropout_rates,
            group_norm_caps=group_norm_caps, cross_layers=cross_layers,
            cross_rank=cross_rank, cross_experts=cross_experts,
            cross_layer_norm=cross_layer_norm, numeric_hidden_dim=numeric_hidden_dim,
            numeric_clip=numeric_clip, se_hidden_dim=se_hidden_dim)
        self.numeric_dim = numeric_dim
        self.numeric_projection_dim = numeric_projection_dim
        self.piecewise_numeric = numeric_encoder
        self.numeric_residual = nn.Linear(numeric_dim, 1, bias=False)
        nn.init.zeros_(self.numeric_residual.weight)

    def forward(self, categories, sequences, numeric):
        clipped = self.piecewise_numeric.clipped_values(numeric)
        projected = self.piecewise_numeric(clipped)
        logits = super().forward(categories, sequences, projected)
        return logits + self.numeric_residual(clipped).squeeze(-1)
