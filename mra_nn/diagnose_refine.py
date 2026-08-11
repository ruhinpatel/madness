#!/usr/bin/env python3
"""Per-level refinement classification diagnostic.

Evaluates the refine head accuracy at each tree level, with focus on
the decision boundary (levels 8-14). Reports precision, recall, F1
per level and per molecule.

Usage:
    python diagnose_refine.py --checkpoint best.pt --config refine_task.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, "/gpfs/projects/rjh/ruhin/madness-ruhin")
sys.path.insert(0, "/gpfs/projects/rjh/adrian/pymra/src")

from mra_nn.dataset import MRADataset
from mra_nn.model import build_model


@torch.no_grad()
def evaluate_refine(model, ds: MRADataset, device: torch.device, batch_size: int = 4096):
    """Evaluate refine head on a dataset. Returns predictions, targets, levels, negative flags."""
    model.eval()
    all_pred_logits = []
    all_targets = []
    all_levels = []
    all_negative = []

    n = len(ds)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = {k: ds.data[k][start:end].to(device) for k in ds.FIELD_NAMES}

        forward_args = [
            batch["rho0_s"], batch["vnuc_s"],
            batch["halo_rho0"], batch["halo_vnuc"],
            batch["level"],
        ]
        if getattr(model, 'use_parent_features', False):
            forward_args.extend([batch["parent_rho0_s"], batch["parent_vnuc_s"]])
        _, _, ref_logit = model(*forward_args)

        all_pred_logits.append(ref_logit.cpu())
        all_targets.append(batch["refine"].cpu())
        all_levels.append(batch["level"].cpu())
        all_negative.append(batch["negative"].cpu())

    logits = torch.cat(all_pred_logits)
    targets = torch.cat(all_targets)
    levels = torch.cat(all_levels)
    negative = torch.cat(all_negative)

    preds = (logits > 0).float()

    return preds, targets, levels, negative


def compute_metrics(preds, targets):
    """Compute precision, recall, F1 for binary classification."""
    tp = ((preds == 1) & (targets == 1)).sum().float()
    fp = ((preds == 1) & (targets == 0)).sum().float()
    fn = ((preds == 0) & (targets == 1)).sum().float()
    tn = ((preds == 0) & (targets == 0)).sum().float()
    precision = float(tp / (tp + fp + 1e-8))
    recall = float(tp / (tp + fn + 1e-8))
    f1 = float(2 * precision * recall / (precision + recall + 1e-8))
    accuracy = float((tp + tn) / (tp + fp + fn + tn + 1e-8))
    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy,
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    h5_path = cfg["data"]["dataset_path"]

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = build_model(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters")

    # --- Val set evaluation ---
    val_mols = cfg["data"]["val_molecules"]
    print(f"\nVal molecules: {val_mols}")

    val_ds = MRADataset(h5_path, val_mols)
    preds, targets, levels, negative = evaluate_refine(model, val_ds, device)

    # Overall metrics (all samples)
    overall = compute_metrics(preds, targets)
    print(f"\n{'='*70}")
    print(f"OVERALL VAL REFINE METRICS (all samples)")
    print(f"{'='*70}")
    print(f"  F1:        {overall['f1']:.4f}")
    print(f"  Precision: {overall['precision']:.4f}")
    print(f"  Recall:    {overall['recall']:.4f}")
    print(f"  Accuracy:  {overall['accuracy']:.4f}")
    print(f"  TP={overall['tp']}  FP={overall['fp']}  FN={overall['fn']}  TN={overall['tn']}")

    # Positive-only metrics (in-tree nodes only)
    pos_mask = negative == 0
    pos_m = compute_metrics(preds[pos_mask], targets[pos_mask])
    print(f"\n{'='*70}")
    print(f"IN-TREE ONLY (negative==0) — where refinement decisions matter")
    print(f"{'='*70}")
    print(f"  F1:        {pos_m['f1']:.4f}")
    print(f"  Precision: {pos_m['precision']:.4f}")
    print(f"  Recall:    {pos_m['recall']:.4f}")
    print(f"  TP={pos_m['tp']}  FP={pos_m['fp']}  FN={pos_m['fn']}  TN={pos_m['tn']}")

    # Per-level breakdown
    print(f"\n{'='*70}")
    print(f"PER-LEVEL BREAKDOWN (all samples)")
    print(f"{'='*70}")
    print(f"  {'Level':>5}  {'Count':>7}  {'Ref=1':>6}  {'Ref=0':>6}  {'Prec':>6}  {'Recall':>6}  {'F1':>6}  {'Acc':>6}")
    print(f"  {'-'*5}  {'-'*7}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}")
    for lvl in sorted(levels.unique().tolist()):
        lmask = levels == lvl
        lm = compute_metrics(preds[lmask], targets[lmask])
        n_ref1 = int(targets[lmask].sum())
        n_ref0 = int((targets[lmask] == 0).sum())
        marker = " <-- boundary" if 8 <= lvl <= 14 else ""
        print(f"  {int(lvl):5d}  {int(lmask.sum()):7d}  {n_ref1:6d}  {n_ref0:6d}  "
              f"{lm['precision']:6.3f}  {lm['recall']:6.3f}  {lm['f1']:6.3f}  {lm['accuracy']:6.3f}{marker}")

    # Per-molecule breakdown
    print(f"\n{'='*70}")
    print(f"PER-MOLECULE REFINE F1 (val)")
    print(f"{'='*70}")
    for mol in val_mols:
        mol_ds = MRADataset(h5_path, [mol])
        mp, mt, ml, mn = evaluate_refine(model, mol_ds, device)
        mol_m = compute_metrics(mp, mt)
        pos_mask = mn == 0
        mol_pos = compute_metrics(mp[pos_mask], mt[pos_mask])
        n_leaves = int((mt == 0).sum())
        n_internal = int((mt == 1).sum())
        print(f"  {mol:15s}: F1={mol_m['f1']:.4f}  (in-tree F1={mol_pos['f1']:.4f})  "
              f"leaves={n_leaves}  internal={n_internal}")

    # Test set
    test_mols = cfg["data"]["test_molecules"]
    print(f"\n{'='*70}")
    print(f"PER-MOLECULE REFINE F1 (test)")
    print(f"{'='*70}")
    for mol in test_mols:
        mol_ds = MRADataset(h5_path, [mol])
        mp, mt, ml, mn = evaluate_refine(model, mol_ds, device)
        mol_m = compute_metrics(mp, mt)
        pos_mask = mn == 0
        mol_pos = compute_metrics(mp[pos_mask], mt[pos_mask])
        print(f"  {mol:15s}: F1={mol_m['f1']:.4f}  (in-tree F1={mol_pos['f1']:.4f})")

    # Gate check
    print(f"\n{'='*70}")
    print(f"GATE CHECK")
    print(f"{'='*70}")
    print(f"  Overall val refine F1: {overall['f1']:.4f}  {'PASS' if overall['f1'] > 0.95 else 'FAIL'} (target > 0.95)")
    print(f"  In-tree val refine F1: {pos_m['f1']:.4f}")


if __name__ == "__main__":
    main()
