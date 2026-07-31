"""Diagnostic: does the model beat rho0 baseline on its own training data?

Loads best.pt checkpoint, runs evaluate() on train and val splits,
prints positive-only rho_s MSE alongside the rho0 baseline for each split.
"""
import sys
import yaml
import torch

sys.path.insert(0, "/gpfs/projects/rjh/adrian/pymra/src")

from dataset import MRADataset
from model import MRANet
from losses import UncertaintyWeightedLoss
from torch.utils.data import DataLoader


def pos_rho_s_mse(model, loader, device):
    """Compute positive-only rho_s MSE for a split."""
    model.eval()
    pred_pos = []
    true_pos = []

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                rs, _, _ = model(
                    batch["rho0_s"], batch["vnuc_s"],
                    batch["halo_rho0"], batch["halo_vnuc"],
                    batch["level"],
                )
            pos_mask = batch["negative"] == 0
            if pos_mask.any():
                pred_pos.append(rs[pos_mask].cpu().float())
                true_pos.append(batch["rho_s"][pos_mask].cpu().float())

    pred = torch.cat(pred_pos)
    true = torch.cat(true_pos)
    return torch.nn.functional.mse_loss(pred, true).item()


def baseline_mse(dataset):
    """rho0-as-is baseline: MSE if we just use promolecular density."""
    pos_mask = dataset.data["negative"] == 0
    rho_s = dataset.data["rho_s"][pos_mask]
    rho0_s = dataset.data["rho0_s"][pos_mask]
    return float((rho_s - rho0_s).pow(2).mean())


def main():
    ckpt_dir = "/gpfs/projects/rjh/ruhin/mra_nn/checkpoints/2026-07-31_04-45"
    ckpt_path = f"{ckpt_dir}/best.pt"

    # Load config from checkpoint
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Checkpoint epoch: {ckpt['epoch']}")
    print()

    # Load datasets
    print("Loading train split...")
    train_ds = MRADataset(data_cfg["dataset_path"], data_cfg["train_molecules"])
    print(f"  {len(train_ds)} samples, {int((train_ds.data['negative'] == 0).sum())} positive")

    print("Loading val split (ch3f)...")
    val_ds = MRADataset(data_cfg["dataset_path"], data_cfg["val_molecules"])
    print(f"  {len(val_ds)} samples, {int((val_ds.data['negative'] == 0).sum())} positive")
    print()

    # Dataloaders (sequential, no sampling)
    train_dl = DataLoader(train_ds, batch_size=4096, shuffle=False, num_workers=4, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=4096, shuffle=False, num_workers=4, pin_memory=True)

    # Build model, load weights
    import inspect
    valid_keys = set(inspect.signature(MRANet.__init__).parameters.keys()) - {"self"}
    model = MRANet(**{k: v for k, v in model_cfg.items() if k in valid_keys}).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Model loaded ({sum(p.numel() for p in model.parameters()):,} params)")
    print()

    # Compute baselines
    train_baseline = baseline_mse(train_ds)
    val_baseline = baseline_mse(val_ds)

    # Compute model MSE on each split
    print("Running model on train split...")
    train_mse = pos_rho_s_mse(model, train_dl, device)

    print("Running model on val split...")
    val_mse = pos_rho_s_mse(model, val_dl, device)

    # Results
    print()
    print("=" * 65)
    print("DIAGNOSTIC: Does the model beat rho0 on its own training data?")
    print("=" * 65)
    print()
    print(f"{'Split':<8} {'Baseline (rho0)':>15} {'Model MSE':>15} {'Ratio':>8} {'Beats?':>8}")
    print("-" * 65)

    train_ratio = train_mse / train_baseline
    val_ratio = val_mse / val_baseline

    train_beats = "YES" if train_mse < train_baseline else "NO"
    val_beats = "YES" if val_mse < val_baseline else "NO"

    print(f"{'Train':<8} {train_baseline:>15.3e} {train_mse:>15.3e} {train_ratio:>7.2f}x {train_beats:>8}")
    print(f"{'Val':<8} {val_baseline:>15.3e} {val_mse:>15.3e} {val_ratio:>7.2f}x {val_beats:>8}")
    print()

    if train_mse < train_baseline and val_mse > val_baseline:
        print("DIAGNOSIS: Overfitting — model learns corrections on train but fails to generalize.")
        print("NEXT: More training data (W4-11 molecules) and/or regularization.")
    elif train_mse > train_baseline and val_mse > val_baseline:
        print("DIAGNOSIS: Can't learn — model doesn't beat rho0 even on training data.")
        print("NEXT: Ablate multi-task heads. Try single-task rho_s-only training.")
    elif train_mse < train_baseline and val_mse < val_baseline:
        print("DIAGNOSIS: Model beats baseline on both splits. Gate 1 should pass.")
        print("NEXT: Run evaluate.py gate check.")
    else:
        print("DIAGNOSIS: Unusual — beats on val but not train. Check for bugs.")


if __name__ == "__main__":
    main()
