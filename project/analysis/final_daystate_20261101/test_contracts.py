"""Exercise actual compact inputs through full-refit training and fresh reload."""
import tempfile
from pathlib import Path
import unittest

import refit_final as runner
import numpy as np
import torch


class FinalRefitContracts(unittest.TestCase):
    def test_uint16_train_step_and_final_only_context_reload(self):
        runner.configure(42)
        meta = runner.read(runner.CACHE / "manifest.json")
        count = 128
        arrays = {
            "categories": np.array(runner.compact_array("categories")[:count], dtype="uint16"),
            "sequences": np.array(runner.compact_array("sequences")[:count], dtype="uint16"),
            "numeric": np.array(runner.compact_array("gate_oof500")[:count, :166]),
        }
        ids = runner.compact_array("ids")[:count]
        raw = runner.pd.DataFrame({"pt_d": ids[:, 2], "user_id": ids[:, 1], "task_id": np.arange(count)})
        # Fit the tiny context scaler on this fixture; production uses train dates only.
        context = runner.build_context(raw, arrays["categories"], arrays["numeric"],
                                       meta["numeric_fields"], meta, runner.FINAL_DAY + 1)
        groups = torch.as_tensor(context["rows"].astype("int64"), device="cuda")
        values = torch.as_tensor(context["values"], device="cuda")
        padded = torch.cat([torch.zeros((2, values.shape[1]), device="cuda"), values], dim=0)
        model = runner.ProtectedUserDayCTR(meta, .0084, padded, .20, use_context=True).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
        actual = runner.inputs(arrays, slice(None))
        self.assertEqual(actual[0].dtype, torch.int64)
        self.assertEqual(actual[1].dtype, torch.int64)
        model.train()
        row, auxiliary = model(*actual, groups + 2)
        labels = torch.zeros(count, device="cuda")
        labels[0] = 1.
        loss = runner.nn.functional.binary_cross_entropy_with_logits(row, labels)
        loss = loss + .02 * runner.auxiliary_loss(auxiliary, labels, torch.ones_like(labels))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        runner.clip_separately(model, model.parameters(), 5.)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            expected = model(*actual, groups + 2)[0].sigmoid().cpu().numpy()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            torch.save(model.state_dict(), path)
            restored = runner.ProtectedUserDayCTR(meta, .0084, values, .20, use_context=True).cuda()
            restored.load_state_dict(torch.load(path, map_location="cpu", weights_only=False), strict=True)
            restored.eval()
            with torch.no_grad():
                replay = restored(*actual, groups)[0].sigmoid().cpu().numpy()
        np.testing.assert_array_equal(replay, expected)
        runner.check_probability(replay, count)


if __name__ == "__main__":
    unittest.main()
