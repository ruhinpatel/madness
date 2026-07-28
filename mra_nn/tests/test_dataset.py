"""Tests for MRA-NN PyTorch dataset."""
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from mra_nn.dataset import MRADataset, build_dataloaders, compute_baseline_mse


DATASET_PATH = "/gpfs/projects/rjh/ruhin/mra_nn/training_dataset.h5"
CONFIG_PATH = str(Path(__file__).resolve().parents[1] / "configs" / "default.yaml")


@pytest.fixture
def cfg():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


@pytest.fixture
def train_ds(cfg):
    return MRADataset(cfg["data"]["dataset_path"], cfg["data"]["train_molecules"])


def test_dataset_length(train_ds):
    assert len(train_ds) > 0
    # 12 train molecules should have ~787k samples
    assert len(train_ds) > 500_000


def test_dataset_getitem_keys(train_ds):
    sample = train_ds[0]
    expected_keys = {
        "rho0_s", "vnuc_s", "halo_rho0", "halo_vnuc",
        "delta_rho", "log_dnorm", "refine", "level", "negative",
    }
    assert set(sample.keys()) == expected_keys


def test_dataset_getitem_shapes(train_ds):
    sample = train_ds[0]
    assert sample["rho0_s"].shape == (216,)
    assert sample["vnuc_s"].shape == (216,)
    assert sample["halo_rho0"].shape == (6, 216)
    assert sample["halo_vnuc"].shape == (6, 216)
    assert sample["delta_rho"].shape == (216,)
    assert sample["log_dnorm"].shape == ()
    assert sample["refine"].shape == ()
    assert sample["level"].shape == ()
    assert sample["negative"].shape == ()


def test_dataset_getitem_dtypes(train_ds):
    sample = train_ds[0]
    assert sample["rho0_s"].dtype == torch.float32
    assert sample["level"].dtype == torch.long
    assert sample["refine"].dtype == torch.float32
    assert sample["negative"].dtype == torch.float32


def test_build_dataloaders(cfg):
    train_dl, val_dl, test_dl = build_dataloaders(cfg)
    batch = next(iter(train_dl))
    assert batch["rho0_s"].shape[0] <= cfg["training"]["batch_size"]
    assert batch["rho0_s"].shape[1] == 216


def test_baseline_mse(train_ds):
    baseline = compute_baseline_mse(train_ds)
    # baseline should be positive (non-zero delta_rho)
    assert baseline > 0.0
    # and finite
    assert np.isfinite(baseline)
