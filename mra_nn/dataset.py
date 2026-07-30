"""PyTorch Dataset wrapping the MRA-NN training HDF5 file.

Loads all samples into memory at init (908k samples, ~10.5 GB — fits on A100).
Provides WeightedRandomSampler for class-imbalanced training.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


class MRADataset(Dataset):
    """In-memory dataset from training_dataset.h5.

    Parameters
    ----------
    h5_path : str
        Path to the training HDF5 file (built by dataset_builder.py).
    molecules : list[str]
        Which molecule groups to include.
    """

    FIELD_NAMES = [
        "rho0_s", "vnuc_s", "halo_rho0", "halo_vnuc",
        "rho_s", "log_dnorm", "refine", "level", "negative",
    ]

    def __init__(self, h5_path: str, molecules: List[str]) -> None:
        arrays: Dict[str, list] = {name: [] for name in self.FIELD_NAMES}
        with h5py.File(h5_path, "r") as f:
            for mol in molecules:
                grp = f[mol]
                for name in self.FIELD_NAMES:
                    arrays[name].append(grp[name][:])

        # Concatenate across molecules
        self.data: Dict[str, torch.Tensor] = {}
        for name in self.FIELD_NAMES:
            arr = np.concatenate(arrays[name], axis=0)
            if name == "level":
                self.data[name] = torch.from_numpy(arr.astype(np.int64))
            elif name in ("refine", "negative"):
                self.data[name] = torch.from_numpy(arr.astype(np.float32))
            else:
                self.data[name] = torch.from_numpy(arr)

        self.n_samples = self.data["level"].shape[0]

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {name: self.data[name][idx] for name in self.FIELD_NAMES}

    def sample_weights(self, refine_pos_weight: float = 10.0) -> torch.Tensor:
        """Per-sample weights for WeightedRandomSampler.

        refine=1 positives get `refine_pos_weight`, everything else gets 1.0.
        """
        w = torch.ones(self.n_samples)
        is_refine_pos = (self.data["refine"] == 1) & (self.data["negative"] == 0)
        w[is_refine_pos] = refine_pos_weight
        return w


def build_dataloaders(cfg: dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build train/val/test DataLoaders from config.

    Train loader uses WeightedRandomSampler for oversampling refine=1 positives.
    Val and test loaders use sequential sampling (no shuffling).
    """
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    loss_cfg = cfg["loss"]

    train_ds = MRADataset(data_cfg["dataset_path"], data_cfg["train_molecules"])
    val_ds = MRADataset(data_cfg["dataset_path"], data_cfg["val_molecules"])
    test_ds = MRADataset(data_cfg["dataset_path"], data_cfg["test_molecules"])

    # Weighted sampler for training
    weights = train_ds.sample_weights(loss_cfg["refine_pos_weight"])
    sampler = WeightedRandomSampler(weights, num_samples=len(train_ds), replacement=True)

    train_dl = DataLoader(
        train_ds,
        batch_size=train_cfg["batch_size"],
        sampler=sampler,
        num_workers=train_cfg["num_workers"],
        pin_memory=True,
        drop_last=True,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=train_cfg["num_workers"],
        pin_memory=True,
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=train_cfg["num_workers"],
        pin_memory=True,
    )
    return train_dl, val_dl, test_dl


def compute_baseline_mse(dataset: MRADataset) -> float:
    """Mean squared error of using rho0_s as the density prediction.

    This is the "use promolecular density as-is" baseline — the model must beat this.
    """
    rho_s = dataset.data["rho_s"]       # [N, 512]
    rho0_s = dataset.data["rho0_s"]     # [N, 512]
    return float((rho_s - rho0_s).pow(2).mean())
