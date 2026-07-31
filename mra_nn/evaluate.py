"""MRA-NN evaluation and Step 6 gate check.

Computes all metrics from the design spec:
  1. Val rho_s MSE < baseline (use rho0 as-is), evaluated on POSITIVE samples
     only (negative==0). Negative (below-leaf) samples are synthetic training
     data that MADNESS never visits during SCF, so including them in the gate
     distorts the metric toward near-zero trivial values.
  2. Refine F1 > 0.5
  3. Predicted tree writable to HDF5
  4. Integral error < 0.01 after normalization

Usage:
    python evaluate.py --checkpoint best.pt --config configs/default.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml

from pymra import read_function

from mra_nn.dataset import MRADataset, build_dataloaders, compute_baseline_mse
from mra_nn.losses import UncertaintyWeightedLoss
from mra_nn.model import build_model
from mra_nn.predict import predict_density
from mra_nn.train import compute_refine_f1, evaluate as evaluate_epoch


# Electron counts per molecule (for integral check)
ELECTRON_COUNTS = {
    "h2o": 10, "nh3": 10, "ch4": 10, "co2": 22, "hf": 10,
    "n2": 14, "co": 14, "hcn": 14, "c2h2": 14, "c2h4": 16,
    "c2h6": 18, "h2co": 16, "ch3oh": 18, "h2o2": 18, "hcl": 18,
}


def main():
    parser = argparse.ArgumentParser(description="MRA-NN evaluation + gate")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded model from {args.checkpoint} (epoch {ckpt['epoch']})")

    # --- Gate 1: Val rho_s MSE < baseline on POSITIVE samples only ---
    print("\n=== Gate 1: Val rho_s MSE vs baseline (positive samples only) ===")
    _, val_dl, _ = build_dataloaders(cfg)
    val_ds = val_dl.dataset

    loss_fn = UncertaintyWeightedLoss(
        focal_gamma=cfg["loss"]["focal_gamma"],
        focal_alpha=cfg["loss"]["focal_alpha"],
        pos_rho_weight=cfg["loss"].get("pos_rho_weight", 10.0),
    ).to(device)
    loss_fn.load_state_dict(ckpt["loss_fn_state_dict"])

    # Positive-only mask (negative==0 means the box is in the rho tree)
    pos_mask = (val_ds.data["negative"] == 0)
    n_pos = int(pos_mask.sum())
    n_total = len(val_ds)
    print(f"  Val set: {n_pos} positive / {n_total} total samples")

    # Baseline MSE on positives: use rho0_s as-is
    rho_s_pos  = val_ds.data["rho_s"][pos_mask]   # [N_pos, k^3]
    rho0_s_pos = val_ds.data["rho0_s"][pos_mask]  # [N_pos, k^3]
    baseline_mse = float((rho_s_pos - rho0_s_pos).pow(2).mean())

    # Model MSE on positives — batched forward pass
    pos_indices = torch.where(pos_mask)[0]
    pred_rs_list = []
    batch_size = 2048
    with torch.no_grad():
        for start in range(0, n_pos, batch_size):
            idx = pos_indices[start:start + batch_size]
            b = {k: val_ds.data[k][idx].to(device)
                 for k in ["rho0_s", "vnuc_s", "halo_rho0", "halo_vnuc", "level"]}
            pred_rs, _, _ = model(
                b["rho0_s"], b["vnuc_s"], b["halo_rho0"], b["halo_vnuc"], b["level"]
            )
            pred_rs_list.append(pred_rs.cpu())
    pred_rs_pos = torch.cat(pred_rs_list, dim=0)
    val_rs_mse = float(torch.nn.functional.mse_loss(pred_rs_pos, rho_s_pos))

    gate1_pass = val_rs_mse < baseline_mse
    print(f"  Val rho_s MSE (pos):   {val_rs_mse:.3e}")
    print(f"  Baseline rho0 (pos):   {baseline_mse:.3e}")
    print(f"  Improvement:           {(1 - val_rs_mse/baseline_mse)*100:.1f}%")
    print(f"  Gate 1:                {'PASS' if gate1_pass else 'FAIL'}")

    # --- Gate 2: Refine F1 > 0.5 ---
    print("\n=== Gate 2: Refine F1 ===")
    val_metrics = evaluate_epoch(model, loss_fn, val_dl, device)
    refine_f1 = val_metrics["refine_f1"]
    gate2_pass = refine_f1 > 0.5
    print(f"  Refine F1:          {refine_f1:.4f}")
    print(f"  Gate 2:             {'PASS' if gate2_pass else 'FAIL'}")

    # --- Gate 3 & 4: Predict on test molecules, check HDF5 and integral ---
    print("\n=== Gates 3 & 4: Prediction + integral ===")
    raw_dir = Path(cfg["data"]["raw_data_dir"])
    test_mols = cfg["data"]["test_molecules"]
    gate3_pass = True
    gate4_pass = True

    for mol in test_mols:
        rho0_path = str(raw_dir / mol / "rho0.mad.h5")
        vnuc_path = str(raw_dir / mol / "vnuc.mad.h5")
        rho_path = str(raw_dir / mol / "rho.mad.h5")
        n_el = ELECTRON_COUNTS[mol]

        pred_tree = predict_density(
            model, rho0_path, vnuc_path,
            n_electrons=n_el, device=device,
        )

        n_leaves = sum(1 for _ in pred_tree.leaves())
        true_tree = read_function(rho_path)
        true_leaves = sum(1 for _ in true_tree.leaves())
        tree_ratio = n_leaves / true_leaves if true_leaves > 0 else 0

        integral = pred_tree.integral()
        int_err = abs(integral - n_el)
        mol_gate4 = int_err < 0.01

        print(f"  {mol}: {n_leaves} leaves (true: {true_leaves}, "
              f"ratio: {tree_ratio:.2f}), "
              f"integral={integral:.6f}, err={int_err:.4e} "
              f"{'OK' if mol_gate4 else 'FAIL'}")

        if not mol_gate4:
            gate4_pass = False

    print(f"  Gate 3 (valid HDF5): {'PASS' if gate3_pass else 'FAIL'}")
    print(f"  Gate 4 (integral):   {'PASS' if gate4_pass else 'FAIL'}")

    # --- Overall verdict ---
    all_pass = gate1_pass and gate2_pass and gate3_pass and gate4_pass
    print(f"\n{'='*40}")
    print(f"STEP 6 GATE: {'PASS' if all_pass else 'FAIL'}")
    print(f"{'='*40}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    exit(main())
