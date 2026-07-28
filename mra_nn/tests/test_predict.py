"""Tests for MRA-NN inference pipeline."""
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from mra_nn.model import build_model
from mra_nn.predict import predict_density

CONFIG_PATH = str(Path(__file__).resolve().parents[1] / "configs" / "default.yaml")
# H2O test data
RHO0_PATH = "/gpfs/projects/rjh/ruhin/mra_nn/training_data/h2o/rho0.mad.h5"
VNUC_PATH = "/gpfs/projects/rjh/ruhin/mra_nn/training_data/h2o/vnuc.mad.h5"
RHO_PATH = "/gpfs/projects/rjh/ruhin/mra_nn/training_data/h2o/rho.mad.h5"
N_ELECTRONS_H2O = 10


@pytest.fixture
def cfg():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


@pytest.fixture
def model(cfg):
    m = build_model(cfg)
    m.eval()
    return m


def test_predict_density_returns_tree(model):
    """predict_density should return a FunctionTree with leaves."""
    from pymra import FunctionTree
    tree = predict_density(
        model, RHO0_PATH, VNUC_PATH,
        n_electrons=N_ELECTRONS_H2O,
        device=torch.device("cpu"),
        max_level=3,
    )
    assert isinstance(tree, FunctionTree)
    leaves = list(tree.leaves())
    assert len(leaves) > 0


def test_predict_density_integral_normalized(model):
    """After post-processing, integral should equal N_electrons."""
    tree = predict_density(
        model, RHO0_PATH, VNUC_PATH,
        n_electrons=N_ELECTRONS_H2O,
        device=torch.device("cpu"),
        max_level=3,
    )
    integral = tree.integral()
    assert abs(integral - N_ELECTRONS_H2O) < 0.01, (
        f"Integral {integral} != {N_ELECTRONS_H2O}"
    )


def test_predict_density_writes_h5(model, tmp_path):
    """Output tree should be writable to HDF5."""
    from pymra import write_function, read_function
    tree = predict_density(
        model, RHO0_PATH, VNUC_PATH,
        n_electrons=N_ELECTRONS_H2O,
        device=torch.device("cpu"),
        max_level=3,
    )
    out_path = str(tmp_path / "predicted_rho.mad.h5")
    write_function(tree, out_path)
    reloaded = read_function(out_path)
    assert len(list(reloaded.leaves())) == len(list(tree.leaves()))
