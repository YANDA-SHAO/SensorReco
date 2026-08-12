from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class GroupedDataset:
    x_sensor: np.ndarray
    y: np.ndarray
    split: np.ndarray
    x_context: np.ndarray | None
    sensor_ids: tuple[int, ...]
    feature_names: tuple[str, ...]

    def validate(self) -> "GroupedDataset":
        x, y, split = np.asarray(self.x_sensor), np.asarray(self.y), np.asarray(self.split).astype(str)
        if x.ndim != 3 or min(x.shape) < 1:
            raise ValueError("x_sensor must have shape [samples, sensors, features]")
        if len(y) != len(x) or len(split) != len(x):
            raise ValueError("x_sensor, y, and split must have the same sample count")
        if not set(np.unique(split)).issubset({"train", "val", "test"}):
            raise ValueError("split values must be train, val, or test")
        if not {"train", "val"}.issubset(set(split)):
            raise ValueError("train and val splits are required")
        if len(self.sensor_ids) != x.shape[1] or len(set(self.sensor_ids)) != len(self.sensor_ids):
            raise ValueError("sensor_ids must uniquely match the sensor axis")
        if self.x_context is not None and (np.asarray(self.x_context).ndim != 2 or len(self.x_context) != len(x)):
            raise ValueError("x_context must have shape [samples, context_features]")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("data contain NaN or infinite values")
        return self


def load_grouped_csv(path: str | Path, *, target: str, sensor_prefix: str = "sensor_", context_prefix: str = "context_") -> GroupedDataset:
    """Load columns named sensor_<id>_<feature> into a grouped tensor."""
    frame = pd.read_csv(path)
    required = {target, "split"}
    if missing := required - set(frame):
        raise ValueError(f"missing required columns: {sorted(missing)}")
    sensor_columns = [c for c in frame if c.startswith(sensor_prefix)]
    parsed = [(c, int(c.split("_")[1]), c.split("_", 2)[2]) for c in sensor_columns]
    sensor_ids = tuple(sorted({p[1] for p in parsed}))
    features = tuple(sorted({p[2] for p in parsed}))
    expected = {f"{sensor_prefix}{s}_{f}" for s in sensor_ids for f in features}
    if set(sensor_columns) != expected:
        raise ValueError("sensor CSV columns do not form a complete sensor-by-feature grid")
    x = np.stack([frame[[f"{sensor_prefix}{s}_{f}" for f in features]].to_numpy() for s in sensor_ids], axis=1).astype(np.float32)
    context_cols = [c for c in frame if c.startswith(context_prefix)]
    context = frame[context_cols].to_numpy(dtype=np.float32) if context_cols else None
    return GroupedDataset(x, frame[target].to_numpy(dtype=np.float32), frame["split"].astype(str).to_numpy(), context, sensor_ids, features).validate()


def load_grouped_npz(path: str | Path) -> GroupedDataset:
    """Load a compact grouped dataset with explicit arrays and metadata."""
    with np.load(path, allow_pickle=False) as z:
        required = {"x_sensor", "y", "split", "sensor_ids", "feature_names"}
        if missing := required - set(z.files):
            raise ValueError(f"missing required NPZ arrays: {sorted(missing)}")
        context = np.asarray(z["x_context"], dtype=np.float32) if "x_context" in z.files else None
        data = GroupedDataset(
            np.asarray(z["x_sensor"], dtype=np.float32),
            np.asarray(z["y"], dtype=np.float32),
            np.asarray(z["split"]).astype(str),
            context,
            tuple(int(v) for v in z["sensor_ids"]),
            tuple(str(v) for v in z["feature_names"]),
        )
    return data.validate()
