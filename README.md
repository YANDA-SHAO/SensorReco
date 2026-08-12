# A Resource-Constrained Sensor Recommendation Framework for AI-Based SHM

This repository provides the research implementation for Gated Sensor Ranking (GSR). 
The release includes GSR and all six comparison procedures used in the
benchmark: Random, QR response diversity, grouped mRMR, Global PFI, High-error
PFI, and retrained leave-one-resource-out (LORO). Raw HBTA data, checkpoints,
cluster scripts, logs, and experiment caches are not distributed.

## Repository layout

```text
configs/       Executable quickstart, reference, and reproduction configurations
data/          Processed Story-7 dataset and machine-readable metadata
scripts/       Benchmark, model-selection, reproduction, and plotting commands
src/gsr/       GSR, Story-7 model, evaluation, and recommendation implementation
src/baselines/ Baseline ranking implementations
tests/         Unit and protocol-contract tests
tools/         Deterministic Story-7 dataset preparation utility
paper_results/ Reserved location for compact publication source-data files
```

## Installation

Python 3.10 or newer is required.

```bash
python -m pip install -e .
python -m pytest -q
```

CUDA is recommended for the paper reproduction commands. Set `"device":
"cpu"` in the selected configuration if a CUDA device is unavailable.

## Quickstart

The quickstart evaluates Random, QR, grouped mRMR, and GSR on representative
sensor budgets using the complete processed dataset:

```bash
python scripts/run_reference_benchmark.py --config configs/quickstart.json
python scripts/plot_recommendation_curves.py
```

Outputs are written to `outputs/quickstart/` and include `rankings.csv`,
`validation_curves.csv`, `recommendations.csv`, and the resolved run
configuration.

To run all seven ranking methods with the compact reference evaluator:

```bash
python scripts/run_reference_benchmark.py --config configs/reference_benchmark.json
```

The reference configuration evaluates budgets `[1, 5, 10, 20, 40, 65]`.
Remove its `budgets` field to evaluate every integer budget from 1 through 65.

## Reproduction

```bash
python scripts/reproduce_story7_benchmark.py --config configs/paper_reproduction.json
```
A shorter protocol check is available with:

```bash
python scripts/reproduce_story7_benchmark.py \
  --config configs/paper_reproduction.json \
  --skip-expensive-baselines
```

The original nine-configuration GSR screening and downstream validation
confirmation can be reproduced independently:

```bash
python scripts/reproduce_gsr_model_selection.py \
  --config configs/paper_reproduction.json
```

## Dataset

`data/story7_balanced.npz` contains the complete processed Story-7 Balanced
task:

- 20,000 samples: 13,991 train, 3,001 validation, and 3,008 test;
- 65 candidate nodes and 15 descriptors per node;
- eight always-available force-context variables;
- the complete 70-dimensional damage target;
- original row order and split membership.

The archive contains `x_sensor`, `x_context`, `y`, `split`, `sensor_ids`,
`feature_names`, `target_names`, and `source_sample_index`. The sensor tensor
uses shape `[samples, sensors, features_per_sensor]`. Context variables are
always available and are excluded from the sensor budget.

## Python API

```python
from gsr import GSRRegressor, select_operating_point

result = GSRRegressor(
    lambda_gsr=1e-4,
    sigma=0.5,
    gate_lr=5e-4,
).fit(
    X_sensor_train,
    y_train,
    X_sensor_validation,
    y_validation,
    context_train,
    context_validation,
)

ranking = result.ranking
operating_point = select_operating_point(
    budgets,
    validation_rmse,
    threshold=0.95,
    higher_is_better=False,
)
```

`gsr.data` validates grouped CSV and NPZ inputs. `gsr.recommendation` provides
the retained-utility operating point, one-standard-error alternative, and
leave-one-training-seed-out sensitivity utilities.
