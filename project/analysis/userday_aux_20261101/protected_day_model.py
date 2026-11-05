"""Keep dense day-label gradients out of the original advertisement backbone."""
from types import SimpleNamespace
import torch
from day_model import UserDayCTR


class ProtectedUserDayCTR(UserDayCTR):
    def forward(self, categories, sequences, numeric, day_rows):
        base = self.backbone
        fields, _, _ = base.encoder(categories, sequences)
        user = torch.tanh(fields[:, self.user_column] / .02)
        context = self.day_context[day_rows]
        row_state = self.day_encoder(torch.cat([context, user], dim=-1))
        auxiliary_state = self.day_encoder(torch.cat([context, user.detach()], dim=-1))
        correction = 2. * self.day_correction(row_state).squeeze(-1).tanh()
        auxiliary = self.any_click(auxiliary_state).squeeze(-1)
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


def clip_separately(model, parameters, max_norm):
    parameters = list(parameters)
    expected = {id(value) for value in model.parameters()}
    if len(parameters) != len(expected) or {id(value) for value in parameters} != expected:
        raise ValueError("Clipping must cover the complete model exactly once")
    backbone_ids = {id(value) for value in model.backbone.parameters()}
    backbone = [value for value in parameters if id(value) in backbone_ids]
    heads = [value for value in parameters if id(value) not in backbone_ids]
    first = torch.nn.utils.clip_grad_norm_(backbone, max_norm)
    second = torch.nn.utils.clip_grad_norm_(heads, max_norm).to(first.device)
    return torch.stack([first, second]).norm()


class NNClippingProxy:
    def __init__(self, module, clip_function):
        self.module = module
        self.utils = SimpleNamespace(clip_grad_norm_=clip_function)

    def __getattr__(self, name):
        return getattr(self.module, name)
