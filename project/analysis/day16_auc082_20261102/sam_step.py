"""A two-pass SAM update with identical dropout masks and exact restoration."""
import math
import torch


def sam_step(model, optimizer, loss_function, rho, clip_function=None):
    if rho < 0 or not math.isfinite(rho):
        raise ValueError("SAM radius must be finite and nonnegative")
    parameters = [p for p in model.parameters() if p.requires_grad]
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    optimizer.zero_grad(set_to_none=True)
    original_loss = loss_function()
    if not bool(torch.isfinite(original_loss)):
        raise ValueError("Nonfinite original objective")
    original_loss.backward()
    gradients = [p.grad for p in parameters if p.grad is not None]
    norm = torch.stack([g.norm() for g in gradients]).norm()
    if not bool(torch.isfinite(norm)):
        raise ValueError("Nonfinite ascent gradient")
    originals = []
    if rho:
        with torch.no_grad():
            for p in parameters:
                if p.grad is not None:
                    originals.append((p,p.detach().clone()))
                    p.add_(p.grad,alpha=rho/max(float(norm),1e-12))
    optimizer.zero_grad(set_to_none=True)
    torch.set_rng_state(cpu_rng)
    if cuda_rng is not None:
        torch.cuda.set_rng_state_all(cuda_rng)
    try:
        perturbed_loss = loss_function()
        if not bool(torch.isfinite(perturbed_loss)):
            raise ValueError("Nonfinite perturbed objective")
        perturbed_loss.backward()
    finally:
        with torch.no_grad():
            for p,old in originals:
                p.copy_(old)
    if clip_function is not None:
        clip_function(parameters)
    optimizer.step()
    return {"original_loss":float(original_loss.detach()),
            "perturbed_loss":float(perturbed_loss.detach()),"ascent_gradient_norm":float(norm)}
