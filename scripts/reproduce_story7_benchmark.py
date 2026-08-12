from __future__ import annotations

import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from baselines import (grouped_mrmr, paper_random_rankings, qr_response_diversity,
                       story7_loro_ranking, story7_permutation_ranking)
from gsr import GSRRegressor, PAPER_GSR_GRID, load_grouped_npz, select_operating_point
from gsr.evaluation import paired_story7_topk_curve


def main():
    ap = argparse.ArgumentParser(description="Complete standalone Story-7 paper protocol")
    ap.add_argument("--config", default="configs/paper_reproduction.json")
    ap.add_argument("--skip-expensive-baselines", action="store_true")
    args = ap.parse_args(); root = Path(__file__).resolve().parents[1]
    cfg = json.loads((root/args.config).read_text()); data = load_grouped_npz(root/cfg["data"])
    out = root/cfg["output_dir"]; out.mkdir(parents=True, exist_ok=True)
    split = np.asarray(data.split); tr = np.flatnonzero(split == "train"); va = np.flatnonzero(split == "val")
    cp, ut = train_test_split(va, test_size=int(round(len(va)*cfg["utility_fraction"])),
                              random_state=cfg["validation_split_seed"], shuffle=True)
    C = data.x_context
    hp = next(x for x in PAPER_GSR_GRID if x.tag == cfg["gsr"]["selected_config"])
    gsr = GSRRegressor(lambda_gsr=hp.lambda_gsr, sigma=hp.sigma, gate_lr=hp.gate_learning_rate,
        lambda_warmup_epochs=hp.lambda_warmup_epochs, lambda_ramp_epochs=hp.lambda_ramp_epochs,
        epochs=hp.epochs, patience=hp.patience, seed=cfg["base_seed"], device=cfg["device"])
    rankings = {"gsr": gsr.fit(data.x_sensor[tr], data.y[tr], data.x_sensor[cp], data.y[cp],
        None if C is None else C[tr], None if C is None else C[cp]).ranking,
        "qr_response_diversity": qr_response_diversity(data.x_sensor[tr]),
        "grouped_mrmr": grouped_mrmr(data.x_sensor[tr], data.y[tr],
            context_train=None if C is None else C[tr],
            redundancy_weight=cfg["baselines"]["mrmr_redundancy_weight"])}
    for rep, ranking in enumerate(paper_random_rankings(data.x_sensor.shape[1],
            base_seed=cfg["base_seed"], repetitions=cfg["baselines"]["random_repetitions"])):
        rankings[f"random_rep_{rep}"] = ranking
    if not args.skip_expensive_baselines:
        common = dict(context_train=None if C is None else C[tr], context_checkpoint=None if C is None else C[cp],
                      context_utility=None if C is None else C[ut], seed=cfg["base_seed"], device=cfg["device"])
        rankings["global_pfi"] = story7_permutation_ranking(data.x_sensor[tr], data.y[tr], data.x_sensor[cp],
            data.y[cp], data.x_sensor[ut], data.y[ut], repeats=cfg["baselines"]["pfi_repetitions"], **common)
        rankings["high_error_pfi"] = story7_permutation_ranking(data.x_sensor[tr], data.y[tr], data.x_sensor[cp],
            data.y[cp], data.x_sensor[ut], data.y[ut], mode="high_error",
            high_error_quantile=cfg["baselines"]["high_error_quantile"],
            repeats=cfg["baselines"]["pfi_repetitions"], **common)
        rankings["loro"] = story7_loro_ranking(data.x_sensor[tr], data.y[tr], data.x_sensor[cp], data.y[cp],
            data.x_sensor[ut], data.y[ut], **common)
    pd.concat([pd.DataFrame({"method": name, "rank": np.arange(1, len(r)+1), "sensor_index": r})
               for name, r in rankings.items()]).to_csv(out/"rankings.csv", index=False)
    budgets = None if cfg["budgets"] == "all" else cfg["budgets"]
    curves, recs = [], []
    for name, ranking in rankings.items():
        print(f"Top-k paired retraining: {name}", flush=True)
        curve = paired_story7_topk_curve(data.x_sensor, data.y, split, ranking, context=C, budgets=budgets,
            base_seed=cfg["base_seed"], retrain_reps=cfg["retrain_repetitions"],
            validation_seed=cfg["validation_split_seed"], utility_fraction=cfg["utility_fraction"], device=cfg["device"])
        curve.insert(0, "method", name); curves.append(curve)
        mean = curve.groupby("budget", as_index=False).validation_rmse.mean()
        op = select_operating_point(mean.budget, mean.validation_rmse, threshold=.95, higher_is_better=False)
        recs.append({"method": name, "k95": op.selected_k, "best_k": op.best_k})
    pd.concat(curves).to_csv(out/"validation_curves_per_seed.csv", index=False)
    pd.DataFrame(recs).to_csv(out/"recommendations.csv", index=False)
    (out/"protocol.json").write_text(json.dumps({**cfg, "test_used_for_selection": False}, indent=2))


if __name__ == "__main__": main()
