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

This root document is the repository's only Markdown file. Dataset provenance,
protocol definitions, usage, and release scope are all documented below.

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

## Paper reproduction

The paper configuration records the validation partition, GSR setting,
baseline protocol, Story-7 MLP, complete budget grid, and paired retraining
design:

```bash
python scripts/reproduce_story7_benchmark.py --config configs/paper_reproduction.json
```

This command uses the fixed validation-selected Story-7 Balanced GSR
configuration `hp07_lam1em4_s0p5_wr_glr5em4`, runs all baselines, evaluates
*k*=1,...,65, and uses five paired retraining seeds. Full LORO first trains the
dense model and 65 leave-one-resource-out models, so the complete command is
computationally expensive. A shorter protocol check is available with:

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

GSR screening uses only the checkpoint portion of validation for early
stopping. Configuration confirmation uses only the disjoint validation-utility
portion and does not evaluate test data. The final benchmark records held-out
test metrics without using them to select the ranking or operating point.

## Fixed experimental protocols

The selected Story-7 Balanced GSR configuration is:

- sparsity coefficient `lambda_gsr = 1e-4`;
- gate noise `sigma = 0.5`;
- predictor learning rate `1e-3`;
- gate learning rate `5e-4`;
- sparsity warm-up of 30 epochs and linear ramp of 70 epochs;
- maximum 500 epochs with early-stopping patience 60.

The Story-7 downstream predictor is a `512-256-128` MLP with batch size 512,
maximum 220 epochs, patience 35, learning rate `1e-3`, weight decay `1e-5`, and
dropout 0.10. Top-*k* evaluation uses the full integer budget grid and five
paired seeds defined as `base_seed + 100000 + repetition`.

Baseline protocol parameters are recorded in `configs/paper_reproduction.json`:

- Random uses five complete allocation permutations with seed
  `base_seed + 50000 + allocation_rep`;
- QR response diversity is deterministic and has no fitted hyperparameter;
- grouped mRMR uses relevance minus mean redundancy with redundancy weight 1;
- Global and High-error PFI use five joint sensor-block permutations;
- High-error PFI fixes the highest-error 20% of validation-utility samples
  before permutation (`high_error_quantile = 0.80`);
- LORO retrains the same Story-7 MLP after removing each resource.

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

Machine-readable provenance is stored in
`data/story7_balanced.metadata.json`. The deterministic preparation
utility is `tools/prepare_story7_dataset.py`.

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

## Publication outputs and release scope

Compact final tables and figure source-data CSV files may be placed in
`paper_results/`. Do not add model checkpoints, caches, raw datasets, or
cluster logs to that directory.

The release is designed to expose the scientific method and the complete
Story-7 example without distributing the manuscript's multi-GPU workspace or
restricted raw HBTA data. Licensing terms are provided in `LICENSE`.
