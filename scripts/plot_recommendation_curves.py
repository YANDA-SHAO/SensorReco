from __future__ import annotations

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot validation budget curves")
    parser.add_argument("--input", default="outputs/quickstart/validation_curves.csv")
    parser.add_argument("--output", default="outputs/quickstart/recommendation_curves.pdf")
    args = parser.parse_args()
    frame = pd.read_csv(args.input)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for method, rows in frame.groupby("method", sort=False):
        ax.plot(rows["budget"], rows["validation_rmse"], marker="o", markersize=3, label=method.replace("_", " "))
    ax.set(xlabel="Sensor budget $k$", ylabel="Validation RMSE")
    ax.grid(True, linestyle="--", alpha=0.4); ax.legend(frameon=False, ncol=2, fontsize=8)
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(output); fig.savefig(output.with_suffix(".png"), dpi=300); plt.close(fig)
    print(f"Figure written to {output}")


if __name__ == "__main__":
    main()
