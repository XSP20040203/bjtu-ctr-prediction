"""Reuse the frozen matched runner with an explicit task-gradient policy."""
import argparse
from pathlib import Path
import run_day_aux as runner
from protected_day_model import ProtectedUserDayCTR, clip_separately, NNClippingProxy

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--day", type=int, choices=[20191115, 20191116], required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag", default="protected_day_aux_v1")
    parser.add_argument("--aux-weight", type=float, default=.02)
    parser.add_argument("--modes", nargs="+", choices=runner.MODES, default=runner.MODES)
    args = parser.parse_args()
    original_model, original_nn, original_write = runner.UserDayCTR, runner.nn, runner.write
    current = {}

    def model_factory(*values, **options):
        model = ProtectedUserDayCTR(*values, **options)
        current["model"] = model
        return model

    def clipping_factory(parameters, max_norm):
        return clip_separately(current["model"], parameters, max_norm)

    def write_protocol(path, value):
        if Path(path).name == "protocol.json":
            value = dict(value)
            value["auxiliary_gradient_policy"] = "Auxiliary day loss updates day encoder/head only; backbone protected"
            value["clipping_policy"] = "Separate backbone and day-head norm caps, each 5"
            value["sources"] = {**value["sources"], **{
                str(HERE / name): runner.digest(HERE / name)
                for name in ["run_protected.py", "protected_day_model.py", "test_protected.py"]}}
        return original_write(path, value)

    runner.UserDayCTR = model_factory
    runner.nn = NNClippingProxy(original_nn, clipping_factory)
    runner.write = write_protocol
    try:
        runner.run(args)
    finally:
        runner.UserDayCTR, runner.nn, runner.write = original_model, original_nn, original_write
        current.clear()


if __name__ == "__main__":
    main()
