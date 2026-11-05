"""Forward compatibility and protection of backbone updates under auxiliary loss."""
import unittest
import torch
from day_model import UserDayCTR, auxiliary_loss
from protected_day_model import ProtectedUserDayCTR, clip_separately
from features import read, cache_dir


class ProtectedTests(unittest.TestCase):
    def models(self, device="cpu"):
        torch.set_num_threads(3)
        meta = read(cache_dir(20191115) / "manifest.json")
        context = torch.ones(4, 7, device=device)
        torch.manual_seed(42)
        original = UserDayCTR(meta, .0084, context, .3, use_context=False).to(device).eval()
        torch.manual_seed(42)
        model = ProtectedUserDayCTR(meta, .0084, context, .3, use_context=False).to(device).eval()
        args = (torch.full((4, 27), 2, dtype=torch.long, device=device),
                torch.full((4, 12, 5), 2, dtype=torch.long, device=device),
                torch.zeros(4, 166, device=device), torch.arange(4, device=device))
        return original, model, args

    def check_route(self, device):
        original, model, args = self.models(device)
        for first, second in zip(original(*args), model(*args)):
            torch.testing.assert_close(first, second, rtol=0, atol=0)
        _, auxiliary = model(*args)
        auxiliary_loss(auxiliary, torch.tensor([1., 0., 1., 0.], device=device),
                        torch.ones(4, device=device)).backward()
        self.assertTrue(all(value.grad is None for value in model.backbone.parameters()))
        self.assertGreater(float(model.day_encoder[0].weight.grad.norm()), 0.)

    def test_cpu_forward_and_gradient_route(self):
        self.check_route("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_forward_and_gradient_route(self):
        self.check_route("cuda")

    def test_auxiliary_only_preserves_backbone_optimization_even_when_clipped(self):
        first, second, args = self.models()
        optimizers = [torch.optim.AdamW(model.parameters(), lr=.001) for model in [first, second]]
        target = torch.tensor([1., 0., 0., 0.])
        auxiliary_target = torch.tensor([1., 1., 0., 0.])
        for _ in range(3):
            for model, optimizer, active in zip([first, second], optimizers, [False, True]):
                optimizer.zero_grad(set_to_none=True)
                logits, aux = model(*args)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
                if active:
                    loss = loss + 10. * auxiliary_loss(aux, auxiliary_target, torch.ones(4))
                loss.backward()
                clip_separately(model, model.parameters(), .01)
                optimizer.step()
            for name, value in first.backbone.state_dict().items():
                torch.testing.assert_close(value, second.backbone.state_dict()[name], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
