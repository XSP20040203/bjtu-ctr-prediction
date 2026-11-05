"""Small, label-free CTR interaction models for the common tensor runner.

ID groups share one embedding table. Histories are unordered, normally five
padded tokens; no history position or recency is inferred. ID magnitude bounds
apply at every lookup, before pooling or attention, without mutating weights.
Total parameters include 8 * sum(vocabulary_sizes.values()) by default, so an
absolute one-million budget also depends on the supplied vocabularies.
"""
from __future__ import annotations

import math
from pathlib import Path
import sys

import torch
from torch import nn

# Support both direct imports from this directory and the common runner.
_comparison_path = str(Path(__file__).resolve().parents[1] / "model_comparison")
if _comparison_path not in sys.path:
    sys.path.insert(0, _comparison_path)

from id_regularized_models import GroupDropoutEncoder
from mainstream_models import dense_network, initial_logit


_DEFAULT_CAPS = {"user_id": .2, "ad_task": .2, "adv_id": .2}
_DEFAULT_QUERIES = {
    "ad_click_list_v001": "task_id",
    "ad_click_list_v002": "adv_prim_id",
    "ad_click_list_v003": "spread_app_id",
    "clicktask": "task_id",
    "clickprim": "adv_prim_id",
    "clickapp": "spread_app_id",
}


class TargetHistoryAttention(nn.Module):
    """Candidate-conditioned attention; padding and empty histories give zero."""

    def __init__(self, embedding_dim, attention_dim=16):
        super().__init__()
        self.query = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.key = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.scale = attention_dim ** -.5

    def forward(self, history, candidate, mask, return_weights=False):
        scores = (self.key(history) * self.query(candidate).unsqueeze(1)).sum(-1) * self.scale
        # A finite sentinel avoids all-masked softmax NaNs, including under AMP.
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1) * mask.to(scores.dtype)
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(torch.finfo(weights.dtype).tiny)
        pooled = (history * weights.unsqueeze(-1)).sum(dim=1)
        return (pooled, weights) if return_weights else pooled


class CappedInteractionEncoder(GroupDropoutEncoder):
    """Shared tables, one dropout draw per read, and differentiable read bounds."""

    def __init__(self, categorical_fields, sequence_fields, group_for_field,
                 vocabulary_sizes, embedding_dim=8, unknown_dropout=.01,
                 group_dropout_rates=None, group_norm_caps=None,
                 history_queries=None, attention_dim=16):
        if not 0 <= unknown_dropout <= 1:
            raise ValueError("unknown_dropout must lie in [0, 1]")
        if embedding_dim <= 0 or not (categorical_fields or sequence_fields):
            raise ValueError("Need a positive embedding dimension and at least one field")
        super().__init__(categorical_fields, sequence_fields, group_for_field,
                         vocabulary_sizes, embedding_dim, unknown_dropout,
                         group_dropout_rates=group_dropout_rates)
        self.embedding_dim = embedding_dim
        self.group_norm_caps = ({g: cap for g, cap in _DEFAULT_CAPS.items() if g in vocabulary_sizes}
                                if group_norm_caps is None else dict(group_norm_caps))
        for group, cap in self.group_norm_caps.items():
            if group not in vocabulary_sizes or not math.isfinite(cap) or cap <= 0:
                raise ValueError(f"Invalid ID norm cap: {group}={cap}")
        self.history_queries = dict(history_queries or {})
        self.query_indices = {}
        for history, candidate in self.history_queries.items():
            if history not in self.sequence_fields or candidate not in self.categorical_fields:
                raise ValueError(f"Unknown history/candidate pair: {history}={candidate}")
            if group_for_field[history] != group_for_field[candidate]:
                raise ValueError(f"History and candidate must share a vocabulary: {history}={candidate}")
            self.query_indices[history] = self.categorical_fields.index(candidate)
        self.history_attention = nn.ModuleDict({
            name: TargetHistoryAttention(embedding_dim, attention_dim)
            for name in self.history_queries
        })

    def read(self, name, values):
        """Return capped vectors and effective indices; never renormalize a table."""
        group = self.group_for_field[name]
        table = self.embeddings[group]
        indices = self._indices(values, table)
        vectors = table(indices)
        if group in self.group_norm_caps:
            norm = vectors.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            vectors = vectors * (self.group_norm_caps[group] / norm).clamp_max(1.)
        # Also makes padding safe after loading a checkpoint with a nonzero PAD row.
        vectors = vectors * values.ne(0).unsqueeze(-1).to(vectors.dtype)
        return vectors, indices

    def forward(self, categories, sequences):
        fields, category_indices, sequence_indices = [], [], []
        for index, name in enumerate(self.categorical_fields):
            vectors, indices = self.read(name, categories[:, index])
            fields.append(vectors)
            category_indices.append(indices)
        for index, name in enumerate(self.sequence_fields):
            original = sequences[:, index]
            vectors, indices = self.read(name, original)
            mask = original.ne(0)
            if name in self.history_attention:
                # Reuse the already-read candidate, including its dropout decision.
                pooled = self.history_attention[name](vectors, fields[self.query_indices[name]], mask)
            else:
                weights = mask.unsqueeze(-1).to(vectors.dtype)
                pooled = (vectors * weights).sum(1) / weights.sum(1).clamp_min(1.)
            fields.append(pooled)
            sequence_indices.append(indices)
        return torch.stack(fields, dim=1), category_indices, sequence_indices


class NumericConditioner(nn.Module):
    def __init__(self, numeric_dim, hidden_dim=16, clip=8.):
        super().__init__()
        if numeric_dim < 0 or hidden_dim <= 0 or not math.isfinite(clip) or clip <= 0:
            raise ValueError("Invalid numeric dimension, conditioning width, or clip")
        self.clip = float(clip)
        self.output_dim = hidden_dim if numeric_dim else 0
        self.projection = nn.Sequential(nn.Linear(numeric_dim, hidden_dim), nn.SiLU()) if numeric_dim else None

    def forward(self, numeric):
        clipped = torch.nan_to_num(numeric, nan=0., posinf=self.clip, neginf=-self.clip).clamp(-self.clip, self.clip)
        summary = self.projection(clipped) if self.projection is not None else clipped
        return clipped, summary


class NumericFieldGate(nn.Module):
    """SE field gates with a compact numeric context; initial gains are near one."""

    def __init__(self, field_count, numeric_summary_dim, hidden_dim=16):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(field_count + numeric_summary_dim, hidden_dim),
                                     nn.SiLU(), nn.Linear(hidden_dim, field_count))
        nn.init.normal_(self.network[-1].weight, std=.01)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, fields, numeric_summary):
        context = torch.cat([fields.mean(-1), numeric_summary], dim=-1)
        gates = 2 * torch.sigmoid(self.network(context))
        return fields * gates.unsqueeze(-1)


class GatedLowRankCross(nn.Module):
    """A content-routed mixture of nonlinear low-rank cross experts.

    x_next = x + x0 * sum_e softmax(router(raw))_e U_e tanh(C_e tanh(V_e x)).
    Optional normalization affects only expert inputs, preserving x0 and the
    residual's original scale instead of normalizing the whole cross output.
    """

    def __init__(self, width, rank=32, experts=3, layer_norm=False):
        super().__init__()
        if rank <= 0 or experts <= 0:
            raise ValueError("rank and experts must be positive")
        self.down = nn.Parameter(torch.empty(experts, width, rank))
        self.core = nn.Parameter(torch.empty(experts, rank, rank))
        self.up = nn.Parameter(torch.empty(experts, rank, width))
        self.expert_bias = nn.Parameter(torch.zeros(experts, width))
        for parameter in (self.down, self.core, self.up):
            for expert in parameter:
                nn.init.xavier_uniform_(expert)
        self.router = nn.Linear(width, experts)
        nn.init.normal_(self.router.weight, std=.01)
        nn.init.zeros_(self.router.bias)
        self.normalizer = nn.LayerNorm(width) if layer_norm else nn.Identity()

    def forward(self, original, current, raw_content):
        low = torch.einsum("bd,edr->ber", self.normalizer(current), self.down).tanh()
        low = torch.einsum("ber,erk->bek", low, self.core).tanh()
        expert_values = torch.einsum("ber,erd->bed", low, self.up) + self.expert_bias
        gates = self.router(raw_content).softmax(-1)
        mixed = torch.einsum("be,bed->bd", gates, expert_values)
        return current + original * mixed


def _small_output(layer, std=.001):
    nn.init.normal_(layer.weight, std=std)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


def _hidden_width(input_width, hidden_dims):
    return hidden_dims[-1] if hidden_dims else input_width


class GatedCrossCTR(nn.Module):
    def __init__(self, categorical_fields, sequence_fields, group_for_field, vocabulary_sizes,
                 numeric_dim=20, initial_ctr=None, embedding_dim=8, hidden_dims=(128, 64),
                 dropout=.1, unknown_dropout=.01, group_dropout_rates=None,
                 group_norm_caps=None, cross_layers=2, cross_rank=32, cross_experts=3,
                 cross_layer_norm=False, numeric_hidden_dim=16, numeric_clip=8.,
                 se_hidden_dim=16):
        super().__init__()
        if cross_layers < 1:
            raise ValueError("cross_layers must be positive")
        self.encoder = CappedInteractionEncoder(
            categorical_fields, sequence_fields, group_for_field, vocabulary_sizes,
            embedding_dim, unknown_dropout, group_dropout_rates, group_norm_caps)
        field_count = len(categorical_fields) + len(sequence_fields)
        self.numeric = NumericConditioner(numeric_dim, numeric_hidden_dim, numeric_clip)
        self.field_gate = NumericFieldGate(field_count, self.numeric.output_dim, se_hidden_dim)
        width = field_count * embedding_dim + numeric_dim
        self.cross = nn.ModuleList([GatedLowRankCross(width, cross_rank, cross_experts,
                                                    cross_layer_norm) for _ in range(cross_layers)])
        self.deep = dense_network(width, hidden_dims, dropout)
        self.output = nn.Linear(width + _hidden_width(width, hidden_dims), 1, bias=False)
        _small_output(self.output)
        self.bias = nn.Parameter(torch.tensor(initial_logit(initial_ctr)))

    def forward(self, categories, sequences, numeric):
        fields, _, _ = self.encoder(categories, sequences)
        clipped, summary = self.numeric(numeric)
        raw_content = torch.cat([fields.flatten(1), clipped], dim=-1)
        original = torch.cat([self.field_gate(fields, summary).flatten(1), clipped], dim=-1)
        crossed = original
        for layer in self.cross:
            crossed = layer(original, crossed, raw_content)
        return self.bias + self.output(torch.cat([crossed, self.deep(original)], dim=-1)).squeeze(-1)


class AutoIntLayer(nn.Module):
    """Multihead field interactions and a learned residual, without a large FFN."""

    def __init__(self, token_dim=16, heads=2, dropout=.1, layer_norm=False):
        super().__init__()
        self.attention = nn.MultiheadAttention(token_dim, heads, dropout=dropout, batch_first=True)
        self.residual = nn.Linear(token_dim, token_dim, bias=False)
        nn.init.eye_(self.residual.weight)
        self.normalizer = nn.LayerNorm(token_dim) if layer_norm else nn.Identity()
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens, padding_mask):
        attended, _ = self.attention(tokens, tokens, tokens, key_padding_mask=padding_mask,
                                     need_weights=False)
        tokens = torch.relu(self.normalizer(self.residual(tokens) + self.dropout(attended)))
        return tokens.masked_fill(padding_mask.unsqueeze(-1), 0.)


class AttentiveInteractionCTR(nn.Module):
    def __init__(self, categorical_fields, sequence_fields, group_for_field, vocabulary_sizes,
                 numeric_dim=20, initial_ctr=None, embedding_dim=8, hidden_dims=(64, 32),
                 dropout=.1, unknown_dropout=.01, group_dropout_rates=None,
                 group_norm_caps=None, token_dim=16, attention_heads=2, attention_layers=2,
                 attention_dim=16, attention_layer_norm=False, numeric_hidden_dim=16,
                 numeric_clip=8., se_hidden_dim=16, history_queries=None):
        super().__init__()
        if token_dim <= 0 or attention_heads <= 0 or token_dim % attention_heads or attention_layers < 1:
            raise ValueError("Need positive attention layers and token_dim divisible by attention_heads")
        if history_queries is None:
            history_queries = {s: c for s, c in _DEFAULT_QUERIES.items()
                               if s in sequence_fields and c in categorical_fields}
        self.encoder = CappedInteractionEncoder(
            categorical_fields, sequence_fields, group_for_field, vocabulary_sizes,
            embedding_dim, unknown_dropout, group_dropout_rates, group_norm_caps,
            history_queries, attention_dim)
        field_count = len(categorical_fields) + len(sequence_fields)
        self.numeric = NumericConditioner(numeric_dim, numeric_hidden_dim, numeric_clip)
        self.field_gate = NumericFieldGate(field_count, self.numeric.output_dim, se_hidden_dim)
        self.field_projection = nn.Linear(embedding_dim, token_dim, bias=False)
        self.field_identity = nn.Parameter(torch.empty(field_count, token_dim))
        nn.init.normal_(self.field_identity, std=.01)
        self.numeric_projection = (nn.Linear(self.numeric.output_dim, token_dim)
                                   if self.numeric.output_dim else None)
        # A numeric-summary token is always unmasked, even for numeric_dim=0.
        self.summary_identity = nn.Parameter(torch.empty(1, 1, token_dim))
        nn.init.normal_(self.summary_identity, std=.01)
        self.interactions = nn.ModuleList([
            AutoIntLayer(token_dim, attention_heads, dropout, attention_layer_norm)
            for _ in range(attention_layers)])
        width = field_count * embedding_dim + numeric_dim
        self.deep = dense_network(width, hidden_dims, dropout)
        self.output = nn.Linear((field_count + 1) * token_dim + _hidden_width(width, hidden_dims),
                                1, bias=False)
        _small_output(self.output)
        self.bias = nn.Parameter(torch.tensor(initial_logit(initial_ctr)))

    def forward(self, categories, sequences, numeric):
        fields, _, _ = self.encoder(categories, sequences)
        clipped, summary = self.numeric(numeric)
        fields = self.field_gate(fields, summary)
        tokens = self.field_projection(fields) + self.field_identity.unsqueeze(0)
        summary_token = self.summary_identity.expand(categories.shape[0], -1, -1)
        if self.numeric_projection is not None:
            summary_token = summary_token + self.numeric_projection(summary).unsqueeze(1)
        tokens = torch.cat([tokens, summary_token], dim=1)
        padding_mask = torch.cat([categories.eq(0), sequences.eq(0).all(dim=-1),
                                  torch.zeros((categories.shape[0], 1), dtype=torch.bool,
                                              device=categories.device)], dim=1)
        tokens = tokens.masked_fill(padding_mask.unsqueeze(-1), 0.)
        for layer in self.interactions:
            tokens = layer(tokens, padding_mask)
        deep_input = torch.cat([fields.flatten(1), clipped], dim=-1)
        return self.bias + self.output(torch.cat([tokens.flatten(1), self.deep(deep_input)], dim=-1)).squeeze(-1)


class ContextFeatureSelection(nn.Module):
    """Independent field and numeric gates, driven by a shared context vector."""

    def __init__(self, field_count, numeric_dim, context_dim, hidden_dim=32):
        super().__init__()
        self.context = nn.Sequential(nn.Linear(context_dim, hidden_dim), nn.SiLU())
        self.field_output = nn.Linear(hidden_dim, field_count)
        self.numeric_output = nn.Linear(hidden_dim, numeric_dim) if numeric_dim else None
        _small_output(self.field_output, std=.01)
        if self.numeric_output is not None:
            _small_output(self.numeric_output, std=.01)

    def forward(self, fields, numeric_features, context):
        hidden = self.context(context)
        field_gates = 2 * torch.sigmoid(self.field_output(hidden))
        numeric_gates = (2 * torch.sigmoid(self.numeric_output(hidden))
                         if self.numeric_output is not None else numeric_features)
        return fields * field_gates.unsqueeze(-1), numeric_features * numeric_gates


class MultiHeadBilinear(nn.Module):
    """Block-diagonal bilinear fusion: sum_h left_h.T W_h right_h / sqrt(d)."""

    def __init__(self, width=64, heads=4):
        super().__init__()
        if width <= 0 or heads <= 0 or width % heads:
            raise ValueError("Bilinear width must be positive and divisible by heads")
        self.heads = heads
        self.head_dim = width // heads
        self.weight = nn.Parameter(torch.empty(heads, self.head_dim, self.head_dim))
        nn.init.normal_(self.weight, std=.001)

    def forward(self, left, right):
        left = left.reshape(-1, self.heads, self.head_dim)
        right = right.reshape(-1, self.heads, self.head_dim)
        scores = torch.einsum("bhi,hij,bhj->bh", left, self.weight, right)
        return scores.sum(-1) / math.sqrt(self.head_dim)


class FinalMLPCTR(nn.Module):
    """FinalMLP-style feature selection, two MLPs, and bilinear block fusion.

    Both independent gates see user context, candidate context and a numeric
    summary. Numeric columns retain their identities: each is robustly squashed
    and multiplied by its own learned scalar embedding, initialized at .02.
    Numeric energy is additionally balanced against the ID feature width. This
    keeps hundreds of standardized columns from overwhelming small ID vectors;
    a separate first-order numeric branch retains clipped numeric magnitudes.

    The linear branch has one scalar table per field, reuses encoder dropout
    decisions, and masked-means history values. Capped groups also have their
    effective linear coefficients bounded, without changing stored weights.
    """

    def __init__(self, categorical_fields, sequence_fields, group_for_field, vocabulary_sizes,
                 numeric_dim=20, initial_ctr=None, embedding_dim=8, hidden_dims=(128, 64),
                 dropout=.1, unknown_dropout=.01, group_dropout_rates=None,
                 group_norm_caps=None, bilinear_heads=4, context_dim=16,
                 gate_hidden_dim=32, numeric_hidden_dim=16, numeric_clip=8.,
                 numeric_feature_scale=.02, user_context_fields=None,
                 candidate_context_fields=None, target_attention=True,
                 history_queries=None, attention_dim=16):
        super().__init__()
        if (not hidden_dims or any(width <= 0 for width in hidden_dims)
                or context_dim <= 0 or gate_hidden_dim <= 0
                or not math.isfinite(numeric_feature_scale) or numeric_feature_scale <= 0):
            raise ValueError("Need positive hidden/context dimensions and numeric feature scale")
        if history_queries is None:
            history_queries = {s: c for s, c in _DEFAULT_QUERIES.items()
                               if s in sequence_fields and c in categorical_fields}
        self.encoder = CappedInteractionEncoder(
            categorical_fields, sequence_fields, group_for_field, vocabulary_sizes,
            embedding_dim, unknown_dropout, group_dropout_rates, group_norm_caps,
            history_queries if target_attention else {}, attention_dim)
        self.field_names = self.encoder.categorical_fields + self.encoder.sequence_fields
        field_count = len(self.field_names)
        self.numeric = NumericConditioner(numeric_dim, numeric_hidden_dim, numeric_clip)
        self.numeric_feature_scale = float(numeric_feature_scale)
        self.numeric_balance = math.sqrt(min(1., field_count * embedding_dim / max(numeric_dim, 1)))
        self.numeric_scale_logits = nn.Parameter(torch.zeros(numeric_dim)) if numeric_dim else None
        candidate_names = {"task_id", "adv_id", "adv_prim_id", "spread_app_id", "slot_id",
                           "site_id", "creat_type_cd", "inter_type_cd", "app_second_class"}
        if candidate_context_fields is None:
            candidate_context_fields = [name for name in self.encoder.categorical_fields
                                        if name in candidate_names]
        if user_context_fields is None:
            user_context_fields = [name for name in self.field_names
                                   if name not in candidate_context_fields]
        for name in list(user_context_fields) + list(candidate_context_fields):
            if name not in self.field_names:
                raise ValueError(f"Unknown context field: {name}")
        self.user_context_indices = [self.field_names.index(name) for name in user_context_fields]
        self.candidate_context_indices = [self.field_names.index(name) for name in candidate_context_fields]
        self.context_dim = context_dim
        self.user_context = (nn.Linear(len(self.user_context_indices) * embedding_dim, context_dim, bias=False)
                             if self.user_context_indices else None)
        self.candidate_context = (nn.Linear(len(self.candidate_context_indices) * embedding_dim, context_dim, bias=False)
                                  if self.candidate_context_indices else None)
        context_width = 2 * context_dim + self.numeric.output_dim
        self.gate_left = ContextFeatureSelection(field_count, numeric_dim, context_width, gate_hidden_dim)
        self.gate_right = ContextFeatureSelection(field_count, numeric_dim, context_width, gate_hidden_dim)
        width = field_count * embedding_dim + numeric_dim
        self.stream_left = dense_network(width, hidden_dims, dropout)
        self.stream_right = dense_network(width, hidden_dims, dropout)
        self.head_left = nn.Linear(hidden_dims[-1], 1, bias=False)
        self.head_right = nn.Linear(hidden_dims[-1], 1, bias=False)
        _small_output(self.head_left)
        _small_output(self.head_right)
        self.bilinear = MultiHeadBilinear(hidden_dims[-1], bilinear_heads)
        self.linear_tables = nn.ModuleDict({
            name: nn.Embedding(vocabulary_sizes[group_for_field[name]], 1, padding_idx=0)
            for name in self.field_names})
        for table in self.linear_tables.values():
            nn.init.zeros_(table.weight)
        self.numeric_linear = nn.Linear(numeric_dim, 1, bias=False) if numeric_dim else None
        if self.numeric_linear is not None:
            _small_output(self.numeric_linear)
        self.bias = nn.Parameter(torch.tensor(initial_logit(initial_ctr)))

    def _numeric_features(self, clipped):
        if self.numeric_scale_logits is None:
            return clipped
        scales = 2 * torch.sigmoid(self.numeric_scale_logits) * self.numeric_feature_scale
        return clipped.tanh() * scales * self.numeric_balance

    def _context(self, fields, summary):
        # Normalize only gate context; keep the actual capped ID feature scales.
        normalized = torch.tanh(fields / .02)
        user = (torch.tanh(self.user_context(normalized[:, self.user_context_indices].flatten(1)))
                if self.user_context is not None else fields.new_zeros(fields.shape[0], self.context_dim))
        candidate = (torch.tanh(self.candidate_context(normalized[:, self.candidate_context_indices].flatten(1)))
                     if self.candidate_context is not None else fields.new_zeros(fields.shape[0], self.context_dim))
        return torch.cat([user, candidate, summary.tanh()], dim=-1)

    def _linear_read(self, name, indices, original):
        values = self.linear_tables[name](indices).squeeze(-1)
        cap = self.encoder.group_norm_caps.get(self.encoder.group_for_field[name])
        if cap is not None:
            values = values.clamp(-cap, cap)
        return values * original.ne(0).to(values.dtype)

    def forward(self, categories, sequences, numeric):
        fields, category_indices, sequence_indices = self.encoder(categories, sequences)
        clipped, summary = self.numeric(numeric)
        numeric_features = self._numeric_features(clipped)
        context = self._context(fields, summary)
        left_fields, left_numeric = self.gate_left(fields, numeric_features, context)
        right_fields, right_numeric = self.gate_right(fields, numeric_features, context)
        left = self.stream_left(torch.cat([left_fields.flatten(1), left_numeric], dim=-1))
        right = self.stream_right(torch.cat([right_fields.flatten(1), right_numeric], dim=-1))
        first_order = (self.numeric_linear(clipped).squeeze(-1) if self.numeric_linear is not None
                       else fields.new_zeros(fields.shape[0]))
        for index, (name, indices) in enumerate(zip(self.encoder.categorical_fields, category_indices)):
            first_order = first_order + self._linear_read(name, indices, categories[:, index])
        for index, (name, indices) in enumerate(zip(self.encoder.sequence_fields, sequence_indices)):
            original = sequences[:, index]
            values = self._linear_read(name, indices, original)
            first_order = first_order + values.sum(-1) / original.ne(0).sum(-1).clamp_min(1)
        return (self.bias + first_order + self.head_left(left).squeeze(-1)
                + self.head_right(right).squeeze(-1) + self.bilinear(left, right))


__all__ = ["GatedCrossCTR", "AttentiveInteractionCTR", "FinalMLPCTR"]
