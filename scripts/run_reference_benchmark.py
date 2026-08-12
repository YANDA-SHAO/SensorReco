from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

from baselines import grouped_mrmr, loro_ranking, permutation_ranking, qr_response_diversity, random_ranking
from gsr import GSRRegressor, load_grouped_csv, load_grouped_npz, select_operating_point
from gsr.evaluation import topk_retrain_curve


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the seven-method grouped-sensor benchmark")
    parser.add_argument("--config", default="configs/reference_benchmark.json")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path = Path(args.config); config_path = config_path if config_path.is_absolute() else root / config_path
    config = json.loads(config_path.read_text(encoding="utf-8"))
    data_path = Path(config["data"]); data_path = data_path if data_path.is_absolute() else root / data_path
    if data_path.suffix.lower() == ".npz":
        data = load_grouped_npz(data_path)
    elif data_path.suffix.lower() == ".csv":
        data = load_grouped_csv(data_path, target=config["target"])
    else:
        raise ValueError("data must be a .csv or .npz file")
    x, y, split, context = data.x_sensor, data.y, data.split, data.x_context
    train, val = split == "train", split == "val"
    requested = set(config.get("methods", ["random", "qr_response_diversity", "grouped_mrmr", "global_pfi", "high_error_pfi", "loro", "gsr"]))
    known = {"random", "qr_response_diversity", "grouped_mrmr", "global_pfi", "high_error_pfi", "loro", "gsr"}
    if unknown := requested - known:
        raise ValueError(f"unknown methods: {sorted(unknown)}")
    g = config["gsr"]
    gsr = GSRRegressor(
        lambda_gsr=g["lambda_gsr"], sigma=g["sigma"], hidden=g["hidden"],
        lr=g["learning_rate"], gate_lr=g["gate_learning_rate"],
        lambda_warmup_epochs=g["lambda_warmup_epochs"],
        lambda_ramp_epochs=g["lambda_ramp_epochs"],
        epochs=g["epochs"], patience=g["patience"], seed=config["seed"], device=config.get("device", "cpu"),
    )
    rankings = {}
    if "random" in requested: rankings["random"] = random_ranking(x.shape[1], seed=config["seed"])
    if "qr_response_diversity" in requested: rankings["qr_response_diversity"] = qr_response_diversity(x[train])
    if "grouped_mrmr" in requested: rankings["grouped_mrmr"] = grouped_mrmr(x[train], y[train], context_train=None if context is None else context[train])
    if "global_pfi" in requested: rankings["global_pfi"] = permutation_ranking(x[train], y[train], x[val], y[val], context_train=None if context is None else context[train], context_val=None if context is None else context[val], repeats=config["pfi_repeats"], seed=config["seed"])
    if "high_error_pfi" in requested: rankings["high_error_pfi"] = permutation_ranking(x[train], y[train], x[val], y[val], context_train=None if context is None else context[train], context_val=None if context is None else context[val], mode="high_error", repeats=config["pfi_repeats"], seed=config["seed"])
    if "loro" in requested: rankings["loro"] = loro_ranking(x[train], y[train], x[val], y[val], context_train=None if context is None else context[train], context_val=None if context is None else context[val])
    if "gsr" in requested: rankings["gsr"] = gsr.fit(x[train], y[train], x[val], y[val], None if context is None else context[train], None if context is None else context[val]).ranking
    ranking_rows, curves, recommendation_rows = [], [], []
    for method, ranking in rankings.items():
        print(f"Evaluating {method}...", flush=True)
        ranking_rows.extend({"method": method, "rank": rank + 1, "sensor_id": data.sensor_ids[pos]} for rank, pos in enumerate(ranking))
        curve = topk_retrain_curve(x, y, split, ranking, context=context, budgets=config.get("budgets")); curve.insert(0, "method", method); curves.append(curve)
        op = select_operating_point(curve.budget, curve.validation_rmse, threshold=config["recommendation_threshold"], higher_is_better=False)
        recommendation_rows.append({"method": method, "threshold": config["recommendation_threshold"], "best_k": op.best_k, "selected_k": op.selected_k, "retention": op.retention})
    out = Path(config["output_dir"]); out = out if out.is_absolute() else root / out; out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(ranking_rows).to_csv(out / "rankings.csv", index=False)
    pd.concat(curves, ignore_index=True).to_csv(out / "validation_curves.csv", index=False)
    pd.DataFrame(recommendation_rows).to_csv(out / "recommendations.csv", index=False)
    (out / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Results written to {out}")


if __name__ == "__main__":
    main()
