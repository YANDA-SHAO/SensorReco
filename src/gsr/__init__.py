"""Minimal Gated Sensor Ranking API."""

from .gates import StochasticSensorGate
from .data import GroupedDataset, load_grouped_csv, load_grouped_npz
from .recommendation import OperatingPoint, select_operating_point
from .selector import GSRRegressor, GSRResult
from .story7 import Story7Protocol, Story7MLP, fit_story7_mlp, PAPER_STORY7_PROTOCOL
from .protocol import GSRConfig, PAPER_GSR_GRID, PAPER_CONFIRMATION_BUDGETS, PAPER_SELECTED_STORY7_BALANCED, PAPER_BASELINE_PROTOCOL

__all__ = ["StochasticSensorGate", "GroupedDataset", "load_grouped_csv", "load_grouped_npz", "GSRRegressor", "GSRResult", "OperatingPoint", "select_operating_point", "Story7Protocol", "Story7MLP", "fit_story7_mlp", "PAPER_STORY7_PROTOCOL", "GSRConfig", "PAPER_GSR_GRID", "PAPER_CONFIRMATION_BUDGETS", "PAPER_SELECTED_STORY7_BALANCED", "PAPER_BASELINE_PROTOCOL"]
