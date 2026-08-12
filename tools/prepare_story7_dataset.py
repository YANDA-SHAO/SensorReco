"""Export the complete processed Story-7 task for the public release."""
from pathlib import Path
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
release = Path(__file__).resolve().parents[1]
source = ROOT / "datasets/story7/balanced/shm_small_tabular_subset.npz"
destination = release / "data/story7_balanced.npz"
metadata_path = release / "data/story7_balanced.metadata.json"

with np.load(source, allow_pickle=True) as z:
    split = z["split"].astype(str)
    indices = np.arange(len(split), dtype=np.int64)
    counts = {name: int(np.sum(split == name)) for name in ("train", "val", "test")}
    sensor_ids = np.arange(65, dtype=int)
    np.savez_compressed(
        destination,
        x_sensor=np.asarray(z["X_node"], dtype=np.float32),
        x_context=np.asarray(z["X_force"], dtype=np.float32),
        y=np.asarray(z["Y"], dtype=np.float32),
        split=np.asarray(split, dtype="<U5"),
        sensor_ids=sensor_ids,
        feature_names=np.asarray([str(v) for v in z["node_feature_names"][:15]], dtype="<U64"),
        target_names=np.asarray([str(v) for v in z["target_names"]], dtype="<U64"),
        source_sample_index=indices,
    )

metadata_path.write_text(json.dumps({
    "description": "Complete processed Story-7 Balanced benchmark task",
    "split_counts": counts,
    "n_candidate_nodes": 65,
    "candidate_node_ids": sensor_ids.tolist(),
    "features_per_node": 15,
    "target_dimensions": 70,
    "always_available_context_dimensions": 8,
    "contains_complete_task_data": True,
    "paper_result_reproduction_with_release_config": True,
}, indent=2), encoding="utf-8")
print(destination)
