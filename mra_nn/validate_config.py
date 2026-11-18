"""Validate an MRA-NN training config YAML before job submission.

Checks all required keys and molecule availability in the HDF5 dataset.
Exits 0 on success, 1 on any error.

Usage:
    python validate_config.py configs/refine_only.yaml
"""
from __future__ import annotations

import sys
from pathlib import Path

import h5py
import yaml


def err(msg: str) -> None:
    print(f"  FAIL  {msg}", file=sys.stderr)


def ok(msg: str) -> None:
    print(f"  OK    {msg}")


def check_keys(d: dict, keys: list[str], section: str) -> list[str]:
    """Return list of missing keys from dict d."""
    missing = [k for k in keys if k not in d]
    for k in missing:
        err(f"[{section}] missing required key: '{k}'")
    return missing


def main(config_path: str) -> int:
    path = Path(config_path)
    if not path.exists():
        err(f"Config file not found: {config_path}")
        return 1

    with open(path) as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        err("Config is empty or not a valid YAML mapping")
        return 1

    failures = 0

    # ── Top-level sections ────────────────────────────────────────────────────
    for section in ("data", "model", "training", "loss", "checkpoint"):
        if section not in cfg:
            err(f"Missing top-level section: '{section}'")
            failures += 1
    if failures:
        return 1  # Can't proceed without sections

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    loss_cfg = cfg["loss"]
    ckpt_cfg = cfg["checkpoint"]

    # ── data ─────────────────────────────────────────────────────────────────
    failures += len(check_keys(data_cfg, ["dataset_path", "train_molecules", "val_molecules", "test_molecules"], "data"))

    # ── model ─────────────────────────────────────────────────────────────────
    failures += len(check_keys(model_cfg, [
        "k_cubed", "n_faces", "n_levels", "level_embed_dim", "face_embed_dim",
        "halo_encoder_hidden", "halo_encoder_out", "trunk_dims", "dropout",
    ], "model"))

    # ── training ──────────────────────────────────────────────────────────────
    failures += len(check_keys(train_cfg, [
        "seed", "batch_size", "num_workers", "lr", "min_lr",
        "weight_decay", "warmup_epochs", "max_epochs", "patience",
    ], "training"))

    # ── loss ──────────────────────────────────────────────────────────────────
    # refine_pos_weight is always required (used in build_dataloaders regardless of mode)
    failures += len(check_keys(loss_cfg, ["refine_pos_weight"], "loss"))

    # focal_gamma/focal_alpha required for refine_only and multi-task modes
    refine_only = cfg.get("refine_only", False)
    single_task = model_cfg.get("single_task", False)
    if refine_only or not single_task:
        failures += len(check_keys(loss_cfg, ["focal_gamma", "focal_alpha"], "loss"))

    # ── checkpoint ────────────────────────────────────────────────────────────
    failures += len(check_keys(ckpt_cfg, ["dir"], "checkpoint"))

    if failures:
        print(f"\n{failures} config error(s) found — fix before submitting.", file=sys.stderr)
        return 1

    ok("All required config keys present")

    # ── HDF5 molecule availability ────────────────────────────────────────────
    h5_path = data_cfg["dataset_path"]
    if not Path(h5_path).exists():
        err(f"HDF5 dataset not found: {h5_path}")
        return 1

    all_mols = (
        list(data_cfg["train_molecules"])
        + list(data_cfg["val_molecules"])
        + list(data_cfg["test_molecules"])
    )
    missing_mols = []
    with h5py.File(h5_path, "r") as f:
        available = set(f.keys())
        for mol in all_mols:
            if mol not in available:
                missing_mols.append(mol)
                err(f"Molecule '{mol}' not found in HDF5")

    if missing_mols:
        print(f"\n{len(missing_mols)} molecule(s) missing from dataset.", file=sys.stderr)
        return 1

    ok(f"All {len(all_mols)} molecules present in HDF5 ({len(data_cfg['train_molecules'])} train / {len(data_cfg['val_molecules'])} val / {len(data_cfg['test_molecules'])} test)")

    # ── Checkpoint dir writable ───────────────────────────────────────────────
    ckpt_dir = Path(ckpt_cfg["dir"])
    if ckpt_dir.exists() and not ckpt_dir.is_dir():
        err(f"Checkpoint path exists but is not a directory: {ckpt_dir}")
        return 1
    # Try to create it if it doesn't exist
    try:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ok(f"Checkpoint dir OK: {ckpt_dir}")
    except OSError as e:
        err(f"Cannot create checkpoint dir {ckpt_dir}: {e}")
        return 1

    # ── Summary ───────────────────────────────────────────────────────────────
    mode = "refine_only" if refine_only else ("single_task" if single_task else "multi_task")
    print(f"\nConfig valid  mode={mode}  config={config_path}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <config.yaml>", file=sys.stderr)
        sys.exit(1)
    sys.exit(main(sys.argv[1]))
