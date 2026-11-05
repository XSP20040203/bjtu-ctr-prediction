"""Initialize a shared input adapter without changing a trained TabM function."""


def initialize_protected_adapter(model, teacher_state):
    if model.hierarchy_mode != "metadata":
        raise ValueError("Protected initialization requires additive metadata mode")
    allowed = ("prior_encoders.", "log_support_temperature.", "support_counts_")
    current = model.state_dict()
    missing = set(current).difference(teacher_state)
    unexpected = set(teacher_state).difference(current)
    if unexpected or any(not name.startswith(allowed) for name in missing):
        raise ValueError(f"Teacher does not exactly cover the existing backbone: missing={sorted(missing)}, unexpected={sorted(unexpected)}")
    model.load_state_dict(teacher_state, strict=False)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith("prior_encoders."))
    # Eval preserves the frozen teacher's deterministic function; autograd still
    # differentiates it with respect to the new shared metadata representations.
    model.eval()
    return {"loaded_base_tensors": len(teacher_state), "trainable_adapter_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "frozen_base_parameters": sum(p.numel() for p in model.parameters() if not p.requires_grad)}
