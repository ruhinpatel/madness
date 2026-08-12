"""MRA-NN training script.

Usage:
    python train.py --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import csv
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml

from mra_nn.dataset import MRADataset, build_dataloaders, compute_baseline_mse
from mra_nn.losses import RefineOnlyLoss, SingleTaskLoss, UncertaintyWeightedLoss
from mra_nn.model import build_model


def compute_refine_f1(
    pred_logits: torch.Tensor, targets: torch.Tensor
) -> float:
    """Compute F1 score for the refine head."""
    preds = (pred_logits > 0).float()
    tp = ((preds == 1) & (targets == 1)).sum().float()
    fp = ((preds == 1) & (targets == 0)).sum().float()
    fn = ((preds == 0) & (targets == 1)).sum().float()
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    return float(2 * precision * recall / (precision + recall + 1e-8))


def train_one_epoch(
    model: torch.nn.Module,
    loss_fn,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    single_task: bool = False,
    refine_only: bool = False,
) -> dict:
    """Train for one epoch. Returns dict of mean losses."""
    model.train()
    accum = {}
    n_batches = 0

    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            forward_args = [
                batch["rho0_s"], batch["vnuc_s"],
                batch["halo_rho0"], batch["halo_vnuc"],
                batch["level"],
            ]
            if "parent_rho0_s" in batch:
                forward_args.extend([batch["parent_rho0_s"], batch["parent_vnuc_s"]])
            rs, ld, ref = model(*forward_args)
            if single_task:
                total_loss, components = loss_fn(batch, rs)
            elif refine_only:
                total_loss, components = loss_fn(batch, ref)
            else:
                total_loss, components = loss_fn(batch, rs, ld, ref)

        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if not single_task and not refine_only:
            # Clamp log_sigma_rs so sigma_rho_s stays <= 1.
            with torch.no_grad():
                loss_fn.log_sigma_rs.clamp_(max=0.0)

        # Accumulate metrics
        for k, v in components.items():
            accum[k] = accum.get(k, 0.0) + v.item()
        accum["total_loss"] = accum.get("total_loss", 0.0) + total_loss.item()
        n_batches += 1

    return {k: v / n_batches for k, v in accum.items()}


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loss_fn,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    single_task: bool = False,
    refine_only: bool = False,
) -> dict:
    """Evaluate on val/test set. Returns dict of mean losses + positive-only rho_s MSE."""
    model.eval()
    accum = {}
    n_batches = 0
    all_ref_logits = []
    all_ref_targets = []
    pred_rs_pos = []
    true_rs_pos = []

    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            forward_args = [
                batch["rho0_s"], batch["vnuc_s"],
                batch["halo_rho0"], batch["halo_vnuc"],
                batch["level"],
            ]
            if "parent_rho0_s" in batch:
                forward_args.extend([batch["parent_rho0_s"], batch["parent_vnuc_s"]])
            rs, ld, ref = model(*forward_args)
            if single_task:
                total_loss, components = loss_fn(batch, rs)
            elif refine_only:
                total_loss, components = loss_fn(batch, ref)
            else:
                total_loss, components = loss_fn(batch, rs, ld, ref)

        for k, v in components.items():
            accum[k] = accum.get(k, 0.0) + v.item()
        accum["total_loss"] = accum.get("total_loss", 0.0) + total_loss.item()
        n_batches += 1

        if not single_task:
            all_ref_logits.append(ref.cpu())
            all_ref_targets.append(batch["refine"].cpu())

        # Accumulate positive (in-tree) samples for positive-only MSE
        pos_mask = (batch["negative"] == 0)
        if pos_mask.any():
            pred_rs_pos.append(rs[pos_mask].cpu().float())
            true_rs_pos.append(batch["rho_s"][pos_mask].cpu().float())

    metrics = {k: v / n_batches for k, v in accum.items()}

    if not single_task:
        # Refine F1
        all_logits = torch.cat(all_ref_logits)
        all_targets = torch.cat(all_ref_targets)
        metrics["refine_f1"] = compute_refine_f1(all_logits, all_targets)

    # Positive-only rho_s MSE — the gate metric
    if pred_rs_pos:
        metrics["pos_rho_s_mse"] = torch.nn.functional.mse_loss(
            torch.cat(pred_rs_pos), torch.cat(true_rs_pos)
        ).item()
    else:
        metrics["pos_rho_s_mse"] = float("inf")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="MRA-NN training")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Seed
    seed = cfg["training"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data
    print("Loading data...")
    train_dl, val_dl, test_dl = build_dataloaders(cfg)
    print(f"  Train: {len(train_dl.dataset)} samples")
    print(f"  Val:   {len(val_dl.dataset)} samples")
    print(f"  Test:  {len(test_dl.dataset)} samples")

    # Baseline
    # Positive-only baseline: MSE between rho0 and rho for in-tree nodes.
    # Matches the gate metric in evaluate.py. All-sample baseline (~5e-8) is
    # dominated by 87% negatives where rho≈rho0≈0 and is not a useful signal.
    pos_mask = (train_dl.dataset.data["negative"] == 0)
    rho_s_pos = train_dl.dataset.data["rho_s"][pos_mask]
    rho0_s_pos = train_dl.dataset.data["rho0_s"][pos_mask]
    baseline_mse = float((rho_s_pos - rho0_s_pos).pow(2).mean())
    print(f"  Baseline MSE (pos, rho0 as-is): {baseline_mse:.3e}")

    # Model
    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters")

    # Loss
    loss_cfg = cfg["loss"]
    single_task = cfg["model"].get("single_task", False)
    refine_focused = cfg.get("refine_focused", False)
    refine_only = cfg.get("refine_only", False)
    if refine_only:
        loss_fn = RefineOnlyLoss(
            focal_gamma=loss_cfg["focal_gamma"],
            focal_alpha=loss_cfg["focal_alpha"],
        ).to(device)
    elif single_task:
        # Compute per-level sample counts from training data for level masking
        level_counts = None
        min_level_samples = loss_cfg.get("min_level_samples", 0)
        if min_level_samples > 0:
            train_levels = train_dl.dataset.data["level"]
            unique, counts = train_levels.unique(return_counts=True)
            level_counts = {int(l): int(c) for l, c in zip(unique, counts)}
            print(f"  Level masking: min_samples={min_level_samples}")
            for lvl in sorted(level_counts):
                status = "✓" if level_counts[lvl] >= min_level_samples else "✗ masked"
                print(f"    level {lvl:2d}: {level_counts[lvl]:6d} samples {status}")
        loss_fn = SingleTaskLoss(
            pos_rho_weight=loss_cfg.get("pos_rho_weight", 10.0),
            level_counts=level_counts,
            min_level_samples=min_level_samples,
        ).to(device)
    else:
        loss_fn = UncertaintyWeightedLoss(
            focal_gamma=loss_cfg["focal_gamma"],
            focal_alpha=loss_cfg["focal_alpha"],
            pos_rho_weight=loss_cfg.get("pos_rho_weight", 10.0),
        ).to(device)

    # Optimizer (includes loss_fn's learnable sigmas for multi-task)
    train_cfg = cfg["training"]
    params = list(model.parameters()) + list(loss_fn.parameters())
    optimizer = torch.optim.AdamW(
        params,
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
    )

    # LR scheduler: linear warmup then cosine decay
    warmup_epochs = train_cfg["warmup_epochs"]
    max_epochs = train_cfg["max_epochs"]

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / (max_epochs - warmup_epochs)
        min_factor = train_cfg["min_lr"] / train_cfg["lr"]
        return min_factor + 0.5 * (1 - min_factor) * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # AMP scaler
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    # Checkpoint directory
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M")
    ckpt_dir = Path(cfg["checkpoint"]["dir"]) / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoints: {ckpt_dir}")

    # Save config copy
    with open(ckpt_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f)

    # CSV logger
    csv_path = ckpt_dir / "metrics.csv"
    if refine_only:
        csv_fields = [
            "epoch", "lr",
            "train_total_loss", "train_loss_refine",
            "val_total_loss", "val_loss_refine", "val_refine_f1",
        ]
    elif single_task:
        csv_fields = [
            "epoch", "lr",
            "train_total_loss", "train_loss_rho_s",
            "val_total_loss", "val_loss_rho_s", "val_pos_rho_s_mse",
        ]
    else:
        csv_fields = [
            "epoch", "lr",
            "train_total_loss", "train_loss_rho_s", "train_loss_log_dnorm",
            "train_loss_refine", "train_sigma_rho_s", "train_sigma_log_dnorm",
            "train_sigma_refine",
            "val_total_loss", "val_loss_rho_s", "val_loss_log_dnorm",
            "val_loss_refine", "val_sigma_rho_s", "val_sigma_log_dnorm",
            "val_sigma_refine", "val_refine_f1", "val_pos_rho_s_mse",
        ]
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
    csv_writer.writeheader()

    # Training loop
    use_f1_gate = refine_only or (refine_focused and not single_task)
    if use_f1_gate:
        best_gate_value = 0.0  # F1, higher is better
    else:
        best_gate_value = float("inf")  # MSE, lower is better
    patience_counter = 0

    print(f"\nTraining for up to {max_epochs} epochs (patience={train_cfg['patience']})...")
    if refine_only:
        print(f"{'Epoch':>5} {'LR':>10} {'TrainLoss':>10} {'ValRefF1':>9} {'Best':>5}")
        print("-" * 45)
    elif single_task:
        print(f"{'Epoch':>5} {'LR':>10} {'TrainLoss':>10} {'PosValMSE':>11} {'Best':>5}")
        print("-" * 50)
    else:
        print(f"{'Epoch':>5} {'LR':>10} {'TrainLoss':>10} {'PosValMSE':>11} {'ValRefF1':>9} {'SigRs':>8} {'Best':>5}")
        print("-" * 67)

    for epoch in range(max_epochs):
        t0 = time.time()

        train_metrics = train_one_epoch(
            model, loss_fn, train_dl, optimizer, scaler, device,
            single_task=single_task, refine_only=refine_only,
        )
        val_metrics = evaluate(model, loss_fn, val_dl, device,
                               single_task=single_task, refine_only=refine_only)

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        # CSV logging
        row = {"epoch": epoch, "lr": f"{current_lr:.6e}"}
        for k, v in train_metrics.items():
            row[f"train_{k}"] = f"{v:.6f}"
        for k, v in val_metrics.items():
            row[f"val_{k}"] = f"{v:.6f}"
        csv_writer.writerow(row)
        csv_file.flush()

        # Checkpointing — gate metric depends on mode.
        if use_f1_gate:
            current_gate = val_metrics.get("refine_f1", 0.0)
            is_best = current_gate > best_gate_value
        else:
            current_gate = val_metrics["pos_rho_s_mse"]
            is_best = current_gate < best_gate_value

        if is_best:
            best_gate_value = current_gate
            patience_counter = 0
        else:
            patience_counter += 1

        ckpt_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_gate_value": best_gate_value,
            "config": cfg,
        }
        if not single_task and not refine_only:
            ckpt_data["loss_fn_state_dict"] = loss_fn.state_dict()

        if is_best:
            torch.save(ckpt_data, ckpt_dir / "best.pt")

        # Always save last
        torch.save(ckpt_data, ckpt_dir / "last.pt")

        dt = time.time() - t0
        best_marker = "*" if is_best else ""
        if refine_only:
            print(
                f"{epoch:5d} {current_lr:10.2e} {train_metrics['total_loss']:10.4f} "
                f"{val_metrics['refine_f1']:9.4f} {best_marker:>5}"
            )
        elif single_task:
            print(
                f"{epoch:5d} {current_lr:10.2e} {train_metrics['total_loss']:10.4f} "
                f"{val_metrics['pos_rho_s_mse']:11.3e} {best_marker:>5}"
            )
        else:
            print(
                f"{epoch:5d} {current_lr:10.2e} {train_metrics['total_loss']:10.4f} "
                f"{val_metrics['pos_rho_s_mse']:11.3e} {val_metrics['refine_f1']:9.4f} "
                f"{val_metrics['sigma_rho_s']:8.4f} {best_marker:>5}"
            )

        # Early stopping
        if patience_counter >= train_cfg["patience"]:
            print(f"\nEarly stopping at epoch {epoch} (patience={train_cfg['patience']})")
            break

    csv_file.close()

    # Final summary
    print(f"\nTraining complete.")
    if use_f1_gate:
        print(f"  Best val refine F1: {best_gate_value:.4f}")
        print(f"  Gate: {'PASS (F1 > 0.95)' if best_gate_value > 0.95 else 'FAIL'}")
    else:
        print(f"  Best pos val rho_s MSE: {best_gate_value:.3e}")
        print(f"  Baseline (pos, rho0):   {baseline_mse:.3e}")
        if best_gate_value < baseline_mse:
            print(f"  Model BEATS baseline by {(1 - best_gate_value/baseline_mse)*100:.1f}%")
        else:
            print(f"  Model DOES NOT beat baseline")
    print(f"  Checkpoints saved to: {ckpt_dir}")


if __name__ == "__main__":
    main()
