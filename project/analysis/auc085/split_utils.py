"""Explicit row partitions; a random split is never reported as a future-day test."""
import numpy as np
from sklearn.model_selection import train_test_split


def make_split(labels, dates, mode, seed=42):
    positions = np.arange(len(labels), dtype=np.int32)
    if mode == "random20":
        train, valid = train_test_split(positions, test_size=.2, random_state=seed, stratify=labels)
        return np.sort(train), np.sort(valid)
    if mode in ["time15", "time16"]:
        day = 20191115 if mode == "time15" else 20191116
        return positions[dates < day], positions[dates == day]
    raise ValueError(f"Unknown validation mode: {mode}")
