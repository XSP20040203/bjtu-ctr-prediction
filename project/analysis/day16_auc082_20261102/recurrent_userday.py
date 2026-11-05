"""Causal recurrent daily response forecasts from earlier observed user days."""
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import argparse
import gc
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from scipy.special import expit
from feed_behavior_userday import prepare, dependency_paths
from neural_userday import preprocess, row_mapping, select_powers, loss_value
from fit_stacker import HERE, ROOT, REFERENCE, digest, write
from rich_userday_activity import array_hash, correct
from verify_stacker import rank_sum_auc

OUT = HERE/"runs/recurrent_userday_v1"
BASE = HERE/"runs/compose_verified_v1"
SELECTOR = HERE/"runs/feed_behavior_userday_v1/binary_15.txt"
SCHEMA = HERE/"runs/feed_behavior_userday_v1/news_schema.json"
SEEDS = [42, 2026]
HIDDEN = [16, 32]


def sequences(values, keys, counts, sizes, cutoff):
    users = np.unique(keys[:, 1])
    by_user = [np.flatnonzero(keys[:, 1] == u) for u in users]
    length = max(map(len, by_user))
    x = np.zeros((len(users), length, values.shape[1]+5), dtype="float32")
    targets = np.full((len(users), length), -1., dtype="float32")
    days = np.zeros((len(users), length), dtype="int32")
    positions = np.full((len(keys), 2), -1, dtype="int64")
    for user_position, rows in enumerate(by_user):
        rows = rows[np.argsort(keys[rows, 0], kind="stable")]
        previous = None
        for time_position, row in enumerate(rows):
            day = int(keys[row, 0])
            x[user_position, time_position, :-5] = values[row]
            if previous is not None:
                prior = previous
                gap = (pd.Timestamp(str(day))-pd.Timestamp(str(int(keys[prior, 0])))).days
                known = keys[prior, 0] < cutoff
                x[user_position, time_position, -5:] = [float(known),
                    np.log1p(counts[prior]) if known else 0., np.log1p(sizes[prior]),
                    counts[prior]/sizes[prior] if known else 0., np.log1p(gap)]
            if day < cutoff:
                if counts[row] < 0:
                    raise ValueError("Earlier daily target is missing")
                targets[user_position, time_position] = counts[row]
            days[user_position, time_position] = day
            positions[row] = [user_position, time_position]
            previous = row
    if np.any(positions < 0) or not np.isfinite(x).all():
        raise AssertionError("Sequence packing failed")
    return x, targets, days, positions


class RecurrentDaily(nn.Module):
    def __init__(self, width, hidden):
        super().__init__()
        self.input = nn.Sequential(nn.Linear(width, 64), nn.SiLU(), nn.Dropout(.15))
        self.recurrent = nn.GRU(64, hidden, batch_first=True)
        self.output = nn.Linear(hidden, 2)

    def forward(self, x):
        z, _ = self.recurrent(self.input(x))
        return self.output(z)


def fit(pack, hidden, seed, fixed_epochs=None):
    x, targets, days, _ = pack
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = RecurrentDaily(x.shape[-1], hidden).cuda()
    tx = torch.as_tensor(x, device="cuda")
    ty = torch.as_tensor(targets, device="cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    valid = torch.as_tensor(days == 20191115, device="cuda")
    best_loss, best_epoch, best_state = np.inf, 0, None
    rng = np.random.default_rng(seed)
    for epoch in range(1, (fixed_epochs or 60)+1):
        model.train()
        order = rng.permutation(len(x))
        for start in range(0, len(order), 512):
            rows = order[start:start+512]
            mask = ty[rows] >= 0
            if not bool(mask.any()):
                continue
            optimizer.zero_grad(set_to_none=True)
            prediction = model(tx[rows])
            loss = loss_value(prediction[mask], ty[rows][mask])
            if not bool(torch.isfinite(loss)):
                raise AssertionError("Nonfinite sequence loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
        if fixed_epochs:
            continue
        model.eval()
        with torch.no_grad():
            # Targets are stored separately from sequence inputs; only the
            # pre15 fit view of the inputs reaches this prediction.
            pred = torch.cat([model(tx[start:start+1024]) for start in range(0, len(x), 1024)])
            held_targets = torch.as_tensor(pack.validation_targets, device="cuda")
            current = float(loss_value(pred[valid], held_targets[valid]))
        if current < best_loss-1e-6:
            best_loss, best_epoch = current, epoch
            best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        if epoch-best_epoch >= 8:
            break
    if fixed_epochs is None:
        model.load_state_dict(best_state)
    return model, fixed_epochs or best_epoch


class SequencePack(tuple):
    pass


def forecast(model, pack, group_rows):
    model.eval()
    x, _, _, positions = pack
    outputs = []
    with torch.no_grad():
        for start in range(0, len(x), 1024):
            z = model(torch.as_tensor(x[start:start+1024], device="cuda"))
            outputs.append(z.cpu().numpy())
    z = np.concatenate(outputs)[tuple(positions[group_rows].T)]
    return np.column_stack([expit(z[:, 0]), np.exp(np.clip(z[:, 1], -9, 5))])


def packs(data, columns, cutoff):
    values = data["x"][:, columns]
    normal = preprocess(values, data["keys"], cutoff)
    values = np.clip((values-normal["center"])/normal["scale"], -8, 8)
    packed = SequencePack(sequences(values, data["keys"], data["counts"], data["sizes"], cutoff))
    validation_targets = np.full(packed[1].shape, -1., dtype="float32")
    mask = data["keys"][:, 0] == 20191115
    validation_targets[tuple(packed[3][mask].T)] = data["counts"][mask]
    packed.validation_targets = validation_targets
    return packed, normal


def deps():
    return list(dict.fromkeys([Path(__file__), HERE/"neural_userday.py", SELECTOR, SCHEMA,
                              BASE/"prediction.npy", BASE/"verification.json"]+dependency_paths()))


def main():
    if OUT.exists():
        raise FileExistsError("Preserving recurrent daily experiment")
    OUT.mkdir(parents=True)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    protocol = {"seeds": SEEDS, "hidden": HIDDEN, "selected_columns": 128,
                "inputs_sha256": {str(p): digest(p) for p in deps()},
                "raw_datasets_sha256": json.loads((HERE/"runs/weighted_v1_day16/protocol.json").read_text(encoding="utf8"))["raw_datasets_sha256"],
                "selection": "Pre15 normalization and labels choose early stop and powers; then pre16 normalization and labels refit",
                "selection_caveat": "Columns use a pre16 daily classifier and are not nested within15 selection;15 is not unbiased validation",
                "causal_inputs": "Current label-free context plus previous observed user-day count, size, rate and actual calendar gap",
                "evaluation16_labels_fit": False, "development16_model_selection": True}
    for name, h in protocol["raw_datasets_sha256"].items():
        if digest(ROOT/"2026ml"/name) != h:
            raise AssertionError("Raw dataset changed")
    write(OUT/"protocol.json", protocol)
    data, _ = prepare(json.loads(SCHEMA.read_text(encoding="utf8")))
    from rich_userday_activity import lgb
    importance = lgb.Booster(model_str=SELECTOR.read_text(encoding="utf8")).feature_importance("gain")
    columns = np.argsort(-importance, kind="stable")[:128]
    pack15, _ = packs(data, columns, 20191115)
    pack16, normal = packs(data, columns, 20191116)
    np.savez(OUT/"normalization16.npz", **normal)
    rows15, groups15, y15 = row_mapping(data, 15)
    base15 = np.load(REFERENCE/"day15_prediction.npy")
    held15 = np.flatnonzero(data["keys"][:, 0] == 20191115)
    locks = []
    for hidden in HIDDEN:
        for seed in SEEDS:
            model, epochs = fit(pack15, hidden, seed)
            gp = forecast(model, pack15, held15)
            powers = select_powers(gp, y15, base15, groups15)
            del model
            final, _ = fit(pack16, hidden, seed, epochs)
            name = f"hidden{hidden}_seed{seed}"
            path = OUT/(name+".pt")
            torch.save(final.state_dict(), path)
            locks.append({"name": name, "hidden": hidden, "seed": seed, "epochs": epochs,
                          "powers": powers, "model_sha256": digest(path)})
            print(json.dumps(locks[-1]), flush=True)
            del final
    write(OUT/"models_locked.json", {"models": locks, "columns": columns.tolist(),
          "normalization_sha256": digest(OUT/"normalization16.npz"), "protocol_sha256": digest(OUT/"protocol.json"),
          "features_sha256": array_hash(data["x"]), "keys_sha256": array_hash(data["keys"]),
          "sequence_sha256": array_hash(pack16[0]), "target_sha256": array_hash(pack16[1])})
    del data, pack15, pack16
    gc.collect()
    torch.cuda.empty_cache()
    evaluate(False)


def evaluate(verifying):
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    protocol = json.loads((OUT/"protocol.json").read_text(encoding="utf8"))
    lock = json.loads((OUT/"models_locked.json").read_text(encoding="utf8"))
    if digest(OUT/"protocol.json") != lock["protocol_sha256"]:
        raise AssertionError("Sequence protocol changed")
    for p, h in protocol["inputs_sha256"].items():
        if digest(p) != h:
            raise AssertionError("Sequence dependency changed")
    for name, h in protocol["raw_datasets_sha256"].items():
        if digest(ROOT/"2026ml"/name) != h:
            raise AssertionError("Raw dataset changed")
    data, _ = prepare(json.loads(SCHEMA.read_text(encoding="utf8")))
    pack, normal = packs(data, np.asarray(lock["columns"]), 20191116)
    if (array_hash(data["x"]) != lock["features_sha256"] or array_hash(data["keys"]) != lock["keys_sha256"]
            or array_hash(pack[0]) != lock["sequence_sha256"] or array_hash(pack[1]) != lock["target_sha256"]):
        raise AssertionError("Causal sequence reconstruction differs")
    saved = np.load(OUT/"normalization16.npz")
    if digest(OUT/"normalization16.npz") != lock["normalization_sha256"]:
        raise AssertionError("Normalization changed")
    for name in normal:
        np.testing.assert_array_equal(normal[name], saved[name])
    held = np.flatnonzero(data["keys"][:, 0] == 20191116)
    rows, groups, y = row_mapping(data, 16)
    if len(rows) != 421983 or int(y.sum()) != 3724 or np.any(pack[1][tuple(pack[3][held].T)] != -1):
        raise AssertionError("Full16 mask contract failed")
    base = np.load(BASE/"prediction.npy")
    records, predictions = [], []
    for r in lock["models"]:
        path = OUT/(r["name"]+".pt")
        if digest(path) != r["model_sha256"]:
            raise AssertionError("Recurrent model changed")
        model = RecurrentDaily(pack[0].shape[-1], r["hidden"]).cuda()
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        gp = forecast(model, pack, held)
        for choice in r["powers"]:
            p = correct(base, gp[:, choice["index"]], groups, choice["objective"], choice["power"])
            records.append({"name": r["name"], **choice, "auc": rank_sum_auc(y, p)})
            predictions.append(p)
        del model
    best = int(np.argmax([r["auc"] for r in records]))
    p = predictions[best]
    error = 0.
    if verifying:
        error = float(np.max(np.abs(p-np.load(OUT/"prediction.npy"))))
        if error > 1e-10:
            raise AssertionError("Recurrent model replay differs")
        np.testing.assert_array_equal(rows, np.load(OUT/"valid_indices.npy"))
    else:
        np.save(OUT/"prediction.npy", p)
        np.save(OUT/"valid_indices.npy", rows)
    result = {"status": "passed" if verifying else "evaluated", "auc": records[best]["auc"],
              "records": records, "rows": len(rows), "clicks": int(y.sum()), "maximum_replay_error": error,
              "original_datasets_unchanged": True, "evaluation16_labels_fit": False,
              "development16_model_selection": True, "target_met": records[best]["auc"] >= .82}
    write(OUT/("verification.json" if verifying else "result.json"), result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    evaluate(True) if parser.parse_args().verify else main()
