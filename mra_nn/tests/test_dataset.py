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

# All tests that open training_dataset.h5 are skipped when the file does not yet exist
# or when it still contains the old schema (delta_rho instead of rho_s).
# The k=8 Slurm data job may still be running.
def _dataset_has_rho_s() -> bool:
    if not Path(DATASET_PATH).exists():
        return False
    try:
        import h5py
        with h5py.File(DATASET_PATH, "r") as f:
            first_mol = next(iter(f.keys()))
            return "rho_s" in f[first_mol]
    except Exception:
        return False

needs_dataset = pytest.mark.skipif(
    not _dataset_has_rho_s(),
    reason=f"training_dataset.h5 with rho_s field not yet available at {DATASET_PATH}",
)


@pytest.fixture
def cfg():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


@pytest.fixture
def train_ds(cfg):
    return MRADataset(cfg["data"]["dataset_path"], cfg["data"]["train_molecules"])


@needs_dataset
def test_dataset_length(train_ds):
    assert len(train_ds) > 0
    # 12 train molecules should have ~787k samples
    assert len(train_ds) > 500_000


@needs_dataset
def test_dataset_getitem_keys(train_ds):
    sample = train_ds[0]
    expected_keys = {
        "rho0_s", "vnuc_s", "halo_rho0", "halo_vnuc",
        "rho_s", "log_dnorm", "refine", "level", "negative",
    }
    assert set(sample.keys()) == expected_keys


@needs_dataset
def test_dataset_getitem_shapes(train_ds):
    sample = train_ds[0]
    assert sample["rho0_s"].shape == (512,)
    assert sample["vnuc_s"].shape == (512,)
    assert sample["halo_rho0"].shape == (6, 512)
    assert sample["halo_vnuc"].shape == (6, 512)
    assert sample["rho_s"].shape == (512,)
    assert sample["log_dnorm"].shape == ()
    assert sample["refine"].shape == ()
    assert sample["level"].shape == ()
    assert sample["negative"].shape == ()


@needs_dataset
def test_dataset_getitem_dtypes(train_ds):
    sample = train_ds[0]
    assert sample["rho0_s"].dtype == torch.float32
    assert sample["level"].dtype == torch.long
    assert sample["refine"].dtype == torch.float32
    assert sample["negative"].dtype == torch.float32


@needs_dataset
def test_build_dataloaders(cfg):
    train_dl, val_dl, test_dl = build_dataloaders(cfg)
    batch = next(iter(train_dl))
    assert batch["rho0_s"].shape[0] <= cfg["training"]["batch_size"]
    assert batch["rho0_s"].shape[1] == 512


@needs_dataset
def test_baseline_mse(train_ds):
    baseline = compute_baseline_mse(train_ds)
    assert baseline > 0.0
    assert np.isfinite(baseline)
