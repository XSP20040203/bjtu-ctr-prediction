"""Invariants for the training-only date adversary."""
import io
import unittest

import torch
from domain_gated_model import DomainGatedCTR, ReverseGradient


class DomainGatedTests(unittest.TestCase):
    def model(self):
        return DomainGatedCTR(['user_id', 'task_id'], ['ad_click_list_v001'],
            {'user_id': 'user_id', 'task_id': 'ad_task', 'ad_click_list_v001': 'ad_task'},
            {'user_id': 8, 'ad_task': 12}, numeric_dim=3, initial_ctr=.0084,
            domain_classes=4, unknown_dropout=0)

    def inputs(self):
        return (torch.tensor([[2, 3], [3, 4], [4, 5]]),
                torch.tensor([[[3, 0, 0]], [[2, 4, 0]], [[0, 0, 0]]]),
                torch.randn(3, 3))

    def test_gradient_reversal_has_exact_sign_and_strength(self):
        values = torch.tensor([1., 2., 3.], requires_grad=True)
        weights = torch.tensor([2., -1., 4.])
        (ReverseGradient.apply(values, .25) * weights).sum().backward()
        torch.testing.assert_close(values.grad, -.25 * weights)

    def test_date_head_does_not_change_inference_logit(self):
        model = self.model().eval()
        inputs = self.inputs()
        logits, date_logits = model.forward_with_domain(*inputs)
        torch.testing.assert_close(logits, model(*inputs), rtol=0, atol=0)
        self.assertEqual(date_logits.shape, (3, 4))
        with torch.no_grad():
            for parameter in model.date_classifier.parameters():
                parameter.fill_(999.)
        torch.testing.assert_close(logits, model(*inputs), rtol=0, atol=0)

    def test_auxiliary_training_gradients_are_finite(self):
        model = self.model()
        logits, date_logits = model.forward_with_domain(*self.inputs())
        loss = (torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.tensor([0., 1., 0.]))
                + .01 * torch.nn.functional.cross_entropy(date_logits, torch.tensor([0, 1, 2])))
        loss.backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
        self.assertGreater(model.date_classifier[-1].weight.grad.abs().sum().item(), 0.)

    def test_checkpoint_replay(self):
        model = self.model().eval()
        inputs = self.inputs()
        expected = model(*inputs)
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        restored = self.model().eval()
        restored.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
        torch.testing.assert_close(expected, restored(*inputs), rtol=0, atol=0)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
