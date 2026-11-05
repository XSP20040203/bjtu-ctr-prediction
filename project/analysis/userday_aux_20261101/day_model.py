"""Zero-initialized day correction and an any-click auxiliary head."""
from pathlib import Path
import math
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis/strong_interest_control_20261031"))
from model import build as build_backbone
import torch
from torch import nn


def auxiliary_loss(logits, targets, group_sizes):
    if bool(((targets < 0) | (targets > 1)).any()) or bool((group_sizes < 1).any()):
        raise ValueError("Auxiliary loss requires observed training targets and positive group sizes")
    weights = group_sizes.float().reciprocal()
    errors = nn.functional.binary_cross_entropy_with_logits(logits, targets.float(), reduction="none")
    return (errors * weights).sum() / weights.sum().clamp_min(1e-12)


class UserDayCTR(nn.Module):
    def __init__(self, meta, prior, contexts, any_prior, use_context=False):
        super().__init__()
        self.backbone = build_backbone(meta, "control", prior)
        self.use_context = use_context
        self.user_column = meta["categorical_fields"].index("user_id")
        self.register_buffer("day_context", contexts, persistent=False)
        width = self.backbone.encoder.embedding_dim
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(92831)
            self.day_encoder = nn.Sequential(nn.Linear(contexts.shape[1] + width, 64), nn.SiLU(),
                                             nn.Linear(64, 32), nn.SiLU())
            self.day_correction = nn.Linear(32, 1)
            self.any_click = nn.Linear(32, 1)
            nn.init.zeros_(self.day_correction.weight)
            nn.init.zeros_(self.day_correction.bias)
            nn.init.normal_(self.any_click.weight, std=.01)
            probability = min(max(any_prior, 1e-5), 1 - 1e-5)
            nn.init.constant_(self.any_click.bias, math.log(probability / (1 - probability)))

    def forward(self, categories, sequences, numeric, day_rows):
        base = self.backbone
        fields, _, _ = base.encoder(categories, sequences)
        user = torch.tanh(fields[:, self.user_column] / .02)
        state = self.day_encoder(torch.cat([self.day_context[day_rows], user], dim=-1))
        correction = 2. * self.day_correction(state).squeeze(-1).tanh()
        auxiliary = self.any_click(state).squeeze(-1)
        clipped, summary = base.numeric(numeric)
        raw_content = torch.cat([fields.flatten(1), clipped], dim=-1)
        original = torch.cat([base.field_gate(fields, summary).flatten(1), clipped], dim=-1)
        crossed = original
        for layer in base.cross:
            crossed = layer(original, crossed, raw_content)
        logit = base.bias + base.output(torch.cat([crossed, base.deep(original)], dim=-1)).squeeze(-1)
        if self.use_context:
            logit = logit + correction
        return logit, auxiliary
