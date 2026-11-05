"""Custom TabM mixture with explicit CIN interactions and a linear expert.

The compressed interaction equations follow the CIN construction in xDeepFM;
the combination with the official TabM backbone and member-wise gate is custom.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from modern_architectures import TabMFieldModel
from bounded_id_models import bound_field_vectors


class CompressedInteraction(nn.Module):
    def __init__(self, n_fields, widths=(32, 32)):
        super().__init__()
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        previous = n_fields
        for width in widths:
            weight = nn.Parameter(torch.empty(width, n_fields, previous))
            nn.init.xavier_normal_(weight)
            self.weights.append(weight)
            self.biases.append(nn.Parameter(torch.zeros(width)))
            previous = width

    def forward(self, original):
        current = original
        outputs = []
        for weight, bias in zip(self.weights, self.biases):
            current = F.relu(torch.einsum("bfd,bhd,ofh->bod", original, current, weight) + bias[None, :, None])
            outputs.append(current.sum(dim=-1))
        return torch.cat(outputs, dim=-1)


class InteractionExpertTabM(TabMFieldModel):
    def __init__(self, *args, cin_widths=(32, 32), **kwargs):
        super().__init__(*args, **kwargs)
        n_fields = len(self.encoder.categorical_fields) + len(self.encoder.sequence_fields)
        width = kwargs.get("d_block", 256)
        numeric_dim = kwargs.get("numeric_dim", 75)
        self.interactions = CompressedInteraction(n_fields, cin_widths)
        self.interaction_head = nn.Linear(sum(cin_widths), 1)
        self.linear_expert = nn.Linear(numeric_dim, 1)
        self.mixture_gate = nn.Linear(width + sum(cin_widths) + numeric_dim, 3)
        nn.init.zeros_(self.linear_expert.weight)
        nn.init.zeros_(self.mixture_gate.weight)
        with torch.no_grad():
            self.mixture_gate.bias.copy_(torch.tensor([2., 0., 0.]))
        initial = kwargs.get("initial_ctr")
        probability = min(max(.5 if initial is None else initial, 1e-6), 1 - 1e-6)
        for head in [self.interaction_head, self.linear_expert]:
            nn.init.constant_(head.bias, math.log(probability / (1 - probability)))

    def forward(self, categories, sequences, numeric):
        fields, _, _ = self.encoder(categories, sequences)
        fields = bound_field_vectors(fields, self.field_groups, self.group_norm_caps)
        dense = numeric if self.numeric_embeddings is None else self.numeric_embeddings(numeric).flatten(1)
        features = torch.cat([fields.flatten(1), dense], dim=-1)
        hidden = self.backbone(features[:, None].expand(-1, self.k, -1))
        normalized = F.normalize(fields.float(), dim=-1, eps=1e-8)
        normalized = bound_field_vectors(normalized, self.field_groups, self.group_norm_caps)
        explicit = self.interactions(normalized)
        deep_logit = self.head(hidden).squeeze(-1)
        interaction_logit = self.interaction_head(explicit).expand(-1, self.k)
        linear_logit = self.linear_expert(numeric).expand(-1, self.k)
        gate_input = torch.cat([hidden, explicit[:, None].expand(-1, self.k, -1), numeric[:, None].expand(-1, self.k, -1)], dim=-1)
        weights = self.mixture_gate(gate_input).float().softmax(dim=-1)
        experts = torch.stack([deep_logit, interaction_logit, linear_logit], dim=-1).float()
        return (weights * experts).sum(dim=-1)
