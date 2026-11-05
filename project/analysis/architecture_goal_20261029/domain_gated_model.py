"""Gated CTR with gradient reversal against training-date predictability.

Date labels are auxiliary optimizer targets from earlier training dates only.
Inference takes exactly the same target-free predictor tensors as the base model.
"""
import torch
from torch import nn
from interaction_models import GatedCrossCTR


class ReverseGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, strength):
        ctx.strength = float(strength)
        return values.view_as(values)

    @staticmethod
    def backward(ctx, gradient):
        return -ctx.strength * gradient, None


class DomainGatedCTR(GatedCrossCTR):
    def __init__(self, *args, domain_classes=5, **options):
        super().__init__(*args, **options)
        if domain_classes < 2:
            raise ValueError('At least two training date domains are required')
        last = self.deep[-1].out_features if isinstance(self.deep[-1], nn.Linear) else None
        if last is None:
            linears = [layer for layer in self.deep if isinstance(layer, nn.Linear)]
            last = linears[-1].out_features
        self.date_classifier = nn.Sequential(nn.Linear(last, 32), nn.SiLU(), nn.Linear(32, domain_classes))
        nn.init.normal_(self.date_classifier[-1].weight, std=.01)
        nn.init.zeros_(self.date_classifier[-1].bias)

    def _forward_parts(self, categories, sequences, numeric):
        fields, _, _ = self.encoder(categories, sequences)
        clipped, summary = self.numeric(numeric)
        raw_content = torch.cat([fields.flatten(1), clipped], dim=-1)
        original = torch.cat([self.field_gate(fields, summary).flatten(1), clipped], dim=-1)
        crossed = original
        for layer in self.cross:
            crossed = layer(original, crossed, raw_content)
        deep = self.deep(original)
        logits = self.bias + self.output(torch.cat([crossed, deep], dim=-1)).squeeze(-1)
        return logits, deep

    def forward(self, categories, sequences, numeric):
        return self._forward_parts(categories, sequences, numeric)[0]

    def forward_with_domain(self, categories, sequences, numeric, reversal_strength=1.):
        logits, representation = self._forward_parts(categories, sequences, numeric)
        dates = self.date_classifier(ReverseGradient.apply(representation, reversal_strength))
        return logits, dates
