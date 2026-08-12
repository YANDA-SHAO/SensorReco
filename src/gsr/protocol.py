from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class GSRConfig:
    tag: str
    lambda_gsr: float
    sigma: float
    gate_learning_rate: float = 1e-3
    mu_init: float = 0.0
    lambda_warmup_epochs: int = 30
    lambda_ramp_epochs: int = 70
    epochs: int = 500
    patience: int = 60

    def to_dict(self): return asdict(self)


PAPER_GSR_GRID = (
    GSRConfig("hp00_ref_lam0_s0p5", 0.0, .5, lambda_warmup_epochs=0, lambda_ramp_epochs=0),
    GSRConfig("hp01_lam3em5_s0p5_wr", 3e-5, .5),
    GSRConfig("hp02_lam1em4_s0p5_nowr", 1e-4, .5, lambda_warmup_epochs=0, lambda_ramp_epochs=0),
    GSRConfig("hp03_lam1em4_s0p5_wr", 1e-4, .5),
    GSRConfig("hp04_lam3em4_s0p5_wr", 3e-4, .5),
    GSRConfig("hp05_lam1em4_s0p3_wr", 1e-4, .3),
    GSRConfig("hp06_lam1em4_s0p8_wr", 1e-4, .8),
    GSRConfig("hp07_lam1em4_s0p5_wr_glr5em4", 1e-4, .5, 5e-4),
    GSRConfig("hp08_lam1em4_s0p5_wr_glr2em3", 1e-4, .5, 2e-3),
)
PAPER_CONFIRMATION_BUDGETS = (1, 3, 5, 10, 20, 40)
PAPER_SELECTED_STORY7_BALANCED = PAPER_GSR_GRID[-2]


@dataclass(frozen=True)
class BaselineProtocol:
    random_repetitions: int = 5
    random_seed_offset: int = 50_000
    pfi_repetitions: int = 5
    high_error_quantile: float = .80
    mrmr_redundancy_weight: float = 1.0


PAPER_BASELINE_PROTOCOL = BaselineProtocol()
