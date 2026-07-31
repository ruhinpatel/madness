"""Quick diagnostic: what is the head actually outputting?

Prints head output norms vs correction norms, and per-level MSE breakdown.
"""
import sys, inspect, torch
sys.path.insert(0, "/gpfs/projects/rjh/adrian/pymra/src")
from dataset import MRADataset
from model import MRANet

ckpt = torch.load(
    "/gpfs/projects/rjh/ruhin/mra_nn/checkpoints/2026-07-31_04-45/best.pt",
    map_location="cpu", weights_only=False,
)
cfg = ckpt["config"]
valid_keys = set(inspect.signature(MRANet.__init__).parameters.keys()) - {"self"}
model = MRANet(**{k: v for k, v in cfg["model"].items() if k in valid_keys})
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

ds = MRADataset(cfg["data"]["dataset_path"], cfg["data"]["train_molecules"])
pos = ds.data["negative"] == 0

rho0 = ds.data["rho0_s"][pos]
vnuc = ds.data["vnuc_s"][pos]
halo_r = ds.data["halo_rho0"][pos]
halo_v = ds.data["halo_vnuc"][pos]
levels = ds.data["level"][pos]
rho_true = ds.data["rho_s"][pos]
delta = rho_true - rho0  # true correction

with torch.no_grad():
    pred, _, _ = model(rho0, vnuc, halo_r, halo_v, levels)
    head_out = pred - rho0  # what the head actually output

print("=== HEAD OUTPUT vs CORRECTION (positive train samples) ===\n")
print(f"  mean |head(x)|:  {head_out.abs().mean():.3e}")
print(f"  mean |Δ|:        {delta.abs().mean():.3e}")
print(f"  ratio:           {head_out.abs().mean() / delta.abs().mean():.2f}x")
print(f"  head RMS:        {head_out.pow(2).mean().sqrt():.3e}")
print(f"  Δ RMS:           {delta.pow(2).mean().sqrt():.3e}")
print()

# Per-level breakdown
print("=== PER-LEVEL: model MSE vs baseline MSE (positive only) ===\n")
print(f"{'Level':>5} {'Count':>6} {'Baseline':>12} {'Model MSE':>12} {'Ratio':>8} {'Beats?':>6}")
print("-" * 55)

for lv in sorted(levels.unique().tolist()):
    mask = levels == lv
    if mask.sum() < 5:
        continue
    lv_delta = delta[mask]
    lv_head = head_out[mask]
    lv_baseline = lv_delta.pow(2).mean().item()
    lv_model = (lv_head - lv_delta).pow(2).mean().item()
    ratio = lv_model / lv_baseline if lv_baseline > 0 else float('inf')
    beats = "YES" if lv_model < lv_baseline else "no"
    print(f"{lv:5d} {mask.sum():6d} {lv_baseline:12.3e} {lv_model:12.3e} {ratio:7.2f}x {beats:>6}")
