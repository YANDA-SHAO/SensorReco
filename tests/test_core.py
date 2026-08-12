import numpy as np
import torch

from gsr import (GSRRegressor, GroupedDataset, PAPER_BASELINE_PROTOCOL,
                 PAPER_GSR_GRID, PAPER_SELECTED_STORY7_BALANCED,
                 PAPER_STORY7_PROTOCOL, StochasticSensorGate, select_operating_point)
from gsr.recommendation import leave_one_seed_out_k, one_standard_error_k


def test_gate_controls_complete_sensor_block():
    gate = StochasticSensorGate(3, sigma=0.5)
    gate.eval()
    with torch.no_grad(): gate.mu[:] = torch.tensor([1.0, -1.0, 0.5])
    x = torch.ones(2, 3, 4)
    y = gate(x)
    assert torch.all(y[:, 0, :] == y[:, 0, :1])
    assert torch.all(y[:, 1, :] == 0)


def test_rmse_operating_point():
    op = select_operating_point([1, 2, 3, 4], [2.0, 1.2, 1.0, 1.05], higher_is_better=False)
    assert op.best_k == 3
    assert op.selected_k == 3


def test_fixed_gsr_penalty_schedule():
    model = GSRRegressor(lambda_warmup_epochs=2, lambda_ramp_epochs=4)
    assert model._lambda_scale(2) == 0.0
    assert model._lambda_scale(4) == 0.5
    assert model._lambda_scale(6) == 1.0


def test_paper_protocol_constants():
    assert len(PAPER_GSR_GRID) == 9
    assert PAPER_SELECTED_STORY7_BALANCED.tag.startswith("hp07_")
    assert PAPER_SELECTED_STORY7_BALANCED.gate_learning_rate == 5e-4
    assert PAPER_STORY7_PROTOCOL.hidden_dims == (512, 256, 128)
    assert PAPER_BASELINE_PROTOCOL.random_repetitions == 5
    assert PAPER_BASELINE_PROTOCOL.high_error_quantile == .80


def test_data_and_sensitivity_contracts():
    data = GroupedDataset(
        np.ones((6, 3, 2)), np.arange(6.0),
        np.array(["train", "train", "train", "val", "val", "test"]),
        np.ones((6, 1)), (0, 1, 2), ("a", "b"),
    ).validate()
    assert data.x_sensor.shape == (6, 3, 2)
    assert one_standard_error_k([1, 2, 3], [0.7, 0.9, 0.91], [0.02, 0.03, 0.06], [5, 5, 5]) == 2
    loo = leave_one_seed_out_k([1, 2, 3], [[0.8, 0.96, 1.0], [0.82, 0.97, 1.0], [0.81, 0.95, 1.0]])
    assert loo.shape == (3,)
