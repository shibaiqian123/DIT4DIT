"""Episode-level Dex1 intent labels. Source files are never modified."""
import json
from pathlib import Path

import numpy as np


def trigger_to_binary(trigger, valid, open_threshold=0.2, close_threshold=0.8):
    """Return float32 [left, right] labels: 0=open, 1=closed.

    Hysteresis is evaluated on the complete episode before window sampling.
    An episode starting in the deadband is initialized at its threshold midpoint.
    Each hand and each episode has independent state.
    """
    if not 0 <= open_threshold < close_threshold <= 1:
        raise ValueError("Require 0 <= open_threshold < close_threshold <= 1")
    trigger = np.asarray(trigger)
    valid = np.asarray(valid)
    if trigger.ndim != 2 or trigger.shape[1] != 2 or len(trigger) == 0:
        raise ValueError("Dex1 trigger must have nonempty shape (T, 2)")
    if valid.shape != trigger.shape or not np.all(valid == 1):
        raise ValueError("Dex1 binary labels require valid triggers for both hands")
    if not np.isfinite(trigger).all() or np.any((trigger < 0) | (trigger > 1)):
        raise ValueError("Dex1 triggers must be finite and in [0, 1]")
    state = trigger[0] >= (open_threshold + close_threshold) / 2
    labels = np.empty(trigger.shape, dtype=np.float32)
    for index, values in enumerate(trigger):
        state = np.where(values >= close_threshold, True,
                         np.where(values <= open_threshold, False, state))
        labels[index] = state
    return labels


def episode_labels(frame, open_threshold=0.2, close_threshold=0.8):
    return trigger_to_binary(np.stack(frame["teleop.dex1_trigger"]),
                             np.stack(frame["dex1.trigger_valid"]),
                             open_threshold, close_threshold)


def label_statistics(root, open_threshold=0.2, close_threshold=0.8):
    """Compute in-memory label statistics; fixed 0/1 bounds also cover constant hands."""
    import pandas as pd

    root = Path(root)
    info = json.loads((root / "meta/info.json").read_text())
    episodes = [json.loads(line) for line in (root / "meta/episodes.jsonl").read_text().splitlines() if line.strip()]
    labels = []
    for episode in episodes:
        index = episode["episode_index"]
        path = root / info["data_path"].format(
            episode_chunk=index // info["chunks_size"], episode_index=index)
        frame = pd.read_parquet(path, columns=["teleop.dex1_trigger", "dex1.trigger_valid"])
        labels.append(episode_labels(frame, open_threshold, close_threshold))
    values = np.concatenate(labels)
    return {key: {"min": [0.0], "max": [1.0],
                  "mean": [float(values[:, side].mean())],
                  "std": [float(values[:, side].std())],
                  "q01": [float(np.quantile(values[:, side], .01))],
                  "q99": [float(np.quantile(values[:, side], .99))]}
            for side, key in enumerate(("left_gripper", "right_gripper"))}
