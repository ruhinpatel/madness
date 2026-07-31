"""Estimate aggregate MSE if we zero out head at levels with <100 samples."""
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
delta = rho_true - rho0

with torch.no_grad():
    pred, _, _ = model(rho0, vnuc, halo_r, halo_v, levels)
    head_out = pred - rho0

# Count samples per level
for min_count in [50, 100, 200, 500]:
    level_counts = {}
    for lv in levels.unique().tolist():
        level_counts[lv] = (levels == lv).sum().item()

    keep_levels = {lv for lv, c in level_counts.items() if c >= min_count}

    # Scenario 1: model as-is (no masking)
    model_mse = (head_out - delta).pow(2).mean().item()
    baseline_mse = delta.pow(2).mean().item()

    # Scenario 2: zero head at low-count levels (use rho0 there)
    masked_head = head_out.clone()
    for lv in levels.unique().tolist():
        if lv not in keep_levels:
            masked_head[levels == lv] = 0.0
    masked_mse = (masked_head - delta).pow(2).mean().item()

    masked_levels = sorted(set(level_counts.keys()) - keep_levels)
    masked_samples = sum(level_counts[lv] for lv in masked_levels)

    print(f"min_count={min_count}: mask levels {masked_levels} ({masked_samples} samples)")
    print(f"  baseline:      {baseline_mse:.3e}")
    print(f"  model as-is:   {model_mse:.3e} ({model_mse/baseline_mse:.3f}x)")
    print(f"  model masked:  {masked_mse:.3e} ({masked_mse/baseline_mse:.3f}x)")
    print(f"  beats baseline? {'YES' if masked_mse < baseline_mse else 'NO'}")
    print()

# Also check: what if we mask on VAL (ch3f)?
print("=== VAL (ch3f) ===\n")
val_ds = MRADataset(cfg["data"]["dataset_path"], cfg["data"]["val_molecules"])
vpos = val_ds.data["negative"] == 0
v_rho0 = val_ds.data["rho0_s"][vpos]
v_vnuc = val_ds.data["vnuc_s"][vpos]
v_halo_r = val_ds.data["halo_rho0"][vpos]
v_halo_v = val_ds.data["halo_vnuc"][vpos]
v_levels = val_ds.data["level"][vpos]
v_rho_true = val_ds.data["rho_s"][vpos]
v_delta = v_rho_true - v_rho0

with torch.no_grad():
    v_pred, _, _ = model(v_rho0, v_vnuc, v_halo_r, v_halo_v, v_levels)
    v_head = v_pred - v_rho0

v_baseline = v_delta.pow(2).mean().item()
v_model = (v_head - v_delta).pow(2).mean().item()

# Use train level counts for masking decision
v_masked_head = v_head.clone()
for lv in v_levels.unique().tolist():
    if level_counts.get(lv, 0) < 100:
        v_masked_head[v_levels == lv] = 0.0

v_masked = (v_masked_head - v_delta).pow(2).mean().item()
print(f"  baseline:      {v_baseline:.3e}")
print(f"  model as-is:   {v_model:.3e} ({v_model/v_baseline:.3f}x)")
print(f"  model masked:  {v_masked:.3e} ({v_masked/v_baseline:.3f}x)")
print(f"  beats baseline? {'YES' if v_masked < v_baseline else 'NO'}")
