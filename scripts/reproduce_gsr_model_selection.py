from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from gsr import GSRRegressor, PAPER_CONFIRMATION_BUDGETS, PAPER_GSR_GRID, load_grouped_npz
from gsr.evaluation import paired_story7_topk_curve


def build_model(cfg, seed, device):
    return GSRRegressor(lambda_gsr=cfg.lambda_gsr, sigma=cfg.sigma,
        gate_lr=cfg.gate_learning_rate, lambda_warmup_epochs=cfg.lambda_warmup_epochs,
        lambda_ramp_epochs=cfg.lambda_ramp_epochs, epochs=cfg.epochs,
        patience=cfg.patience, seed=seed, device=device)


def main():
    p = argparse.ArgumentParser(description="Run the paper's optional nine-config GSR screen and confirmation")
    p.add_argument("--config", default="configs/paper_reproduction.json")
    args = p.parse_args(); root = Path(__file__).resolve().parents[1]
    cfg = json.loads((root/args.config).read_text()); data = load_grouped_npz(root/cfg["data"])
    split = np.asarray(data.split); tr = np.flatnonzero(split == "train"); va = np.flatnonzero(split == "val")
    cp, _ = train_test_split(va, test_size=int(round(len(va)*cfg["utility_fraction"])),
                             random_state=cfg["validation_split_seed"], shuffle=True)
    out = root/cfg["output_dir"]/"gsr_screening"; out.mkdir(parents=True, exist_ok=True)
    summaries, rankings = [], {}
    for hp in PAPER_GSR_GRID:
        print(f"Training {hp.tag}", flush=True)
        result = build_model(hp, cfg["base_seed"], cfg["device"]).fit(
            data.x_sensor[tr], data.y[tr], data.x_sensor[cp], data.y[cp],
            None if data.x_context is None else data.x_context[tr],
            None if data.x_context is None else data.x_context[cp])
        rankings[hp.tag] = result.ranking
        pd.DataFrame({"rank": np.arange(1, len(result.ranking)+1), "sensor_index": result.ranking,
                      "gate_probability": result.probabilities[result.ranking]}).to_csv(out/f"{hp.tag}_ranking.csv", index=False)
        curve = paired_story7_topk_curve(data.x_sensor, data.y, split, result.ranking,
            context=data.x_context, budgets=PAPER_CONFIRMATION_BUDGETS,
            base_seed=cfg["base_seed"], retrain_reps=cfg["gsr"]["confirmation_retrain_repetitions"],
            validation_seed=cfg["validation_split_seed"], utility_fraction=cfg["utility_fraction"],
            device=cfg["device"], evaluate_test=False)
        curve.insert(0, "config_tag", hp.tag); curve.to_csv(out/f"{hp.tag}_confirmation.csv", index=False)
        summaries.append({"config_tag": hp.tag, "mean_confirmation_rmse": curve.validation_rmse.mean()})
    summary = pd.DataFrame(summaries).sort_values(["mean_confirmation_rmse", "config_tag"])
    summary.to_csv(out/"confirmation_summary.csv", index=False)
    selected = summary.iloc[0].config_tag
    (out/"selected_config.json").write_text(json.dumps({"selected_config": selected, "test_used": False}, indent=2))
    print(f"Selected {selected}")


if __name__ == "__main__": main()
