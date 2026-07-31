# MRA-NN Single-Task Ablation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Add a `single_task` mode that removes the log_dnorm and refine heads, trains rho_s only with plain weighted MSE loss, and determines whether multi-task interference is preventing the model from learning density corrections.

**Architecture:** Same halo encoder + FiLM trunk as multi-task, but only the rho_s output head (Linear(256,512) + rho0_s residual). Loss is weighted MSE with 10x positive upweight — no uncertainty weighting, no focal loss, no learnable sigma parameters.

**Tech Stack:** Python 3.12, PyTorch 2.13, pyyaml, h5py, numpy

## Global Constraints

- **Branch:** `feat/mra-nn-data` on `ruhinpatel/madness`
- **All code lives in:** `/gpfs/projects/rjh/ruhin/madness-ruhin/mra_nn/`
- **Training dataset:** `/gpfs/projects/rjh/ruhin/mra_nn/training_dataset.h5`
- **pymra:** `/gpfs/projects/rjh/adrian/pymra/src` — must be on PYTHONPATH
- **Venv:** `/gpfs/projects/rjh/ruhin/mra_nn/.venv/`
- **Checkpoints:** `/gpfs/projects/rjh/ruhin/mra_nn/checkpoints/<run_name>/`
- **Slurm:** `/cm/shared/apps/slurm/21.08.8/bin/sbatch`; A100 partition
- **Git:** No `Co-Authored-By:` lines. No Jira ticket names.
- **Design spec:** `docs/superpowers/specs/2026-07-31-mra-nn-single-task-ablation-design.md`
- **Tests run from repo root:** `cd /gpfs/projects/rjh/ruhin/madness-ruhin && python -m pytest mra_nn/tests/ -v`

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `mra_nn/model.py` | Modify | Add `single_task` param; conditionally skip log_dnorm/refine heads; return `(rho_s, None, None)` in single-task mode |
| `mra_nn/losses.py` | Modify | Add `SingleTaskLoss` class (weighted MSE, no learnable params) |
| `mra_nn/train.py` | Modify | Branch on `single_task` config to use `SingleTaskLoss`, skip sigma clamping/refine metrics, adjust CSV/printing |
| `mra_nn/configs/single_task.yaml` | Create | Config with `single_task: true`, trimmed loss section |
| `mra_nn/tests/test_model.py` | Modify | Add tests for single-task model output shapes |
| `mra_nn/tests/test_losses.py` | Modify | Add tests for `SingleTaskLoss` |

---

### Task 1: Add single-task mode to model and loss

**Files:**
- Modify: `mra_nn/model.py:128-237` (MRANet class + build_model)
- Modify: `mra_nn/losses.py` (add SingleTaskLoss after UncertaintyWeightedLoss)
- Modify: `mra_nn/tests/test_model.py` (add single-task tests)
- Modify: `mra_nn/tests/test_losses.py` (add SingleTaskLoss tests)

**Interfaces:**
- Produces: `MRANet(single_task=True)` — returns `(rho_s, None, None)` from `forward()`
- Produces: `SingleTaskLoss(pos_rho_weight=10.0)` — `forward(batch, pred_rho_s)` returns `(loss, {"loss_rho_s": detached})`
- Produces: `build_model(cfg)` handles `single_task` key in config

- [ ] **Step 1: Write failing tests for single-task model**

Add to `mra_nn/tests/test_model.py`:

```python
def test_single_task_forward_shapes():
    """Single-task model returns (rho_s, None, None)."""
    model = MRANet(single_task=True)
    B = 8
    rho0_s = torch.randn(B, 512)
    vnuc_s = torch.randn(B, 512)
    halo_rho0 = torch.randn(B, 6, 512)
    halo_vnuc = torch.randn(B, 6, 512)
    level = torch.randint(0, 19, (B,))

    rho_s, log_dnorm, refine_logit = model(
        rho0_s, vnuc_s, halo_rho0, halo_vnuc, level
    )
    assert rho_s.shape == (B, 512)
    assert log_dnorm is None
    assert refine_logit is None


def test_single_task_no_extra_heads():
    """Single-task model should not have log_dnorm or refine head parameters."""
    model = MRANet(single_task=True)
    assert not hasattr(model, "head_log_dnorm")
    assert not hasattr(model, "head_refine")
    assert hasattr(model, "head_rho_s")


def test_single_task_parameter_count():
    """Single-task model should have fewer parameters (no log_dnorm/refine heads)."""
    model_full = MRANet(single_task=False)
    model_st = MRANet(single_task=True)
    full_params = sum(p.numel() for p in model_full.parameters())
    st_params = sum(p.numel() for p in model_st.parameters())
    assert st_params < full_params
    # Difference should be exactly: log_dnorm head (256+1=257) + refine head (256+1=257) = 514
    assert full_params - st_params == 514


def test_single_task_build_model(cfg):
    """build_model with single_task config should produce single-task model."""
    cfg["model"]["single_task"] = True
    model = build_model(cfg)
    assert not hasattr(model, "head_log_dnorm")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /gpfs/projects/rjh/ruhin/madness-ruhin && python -m pytest mra_nn/tests/test_model.py -v -k "single_task"`

Expected: FAIL — `MRANet.__init__() got an unexpected keyword argument 'single_task'`

- [ ] **Step 3: Implement single_task mode in MRANet**

In `mra_nn/model.py`, modify `MRANet.__init__` (line 128) to accept `single_task: bool = False`:

```python
def __init__(
    self,
    k_cubed: int = 512,
    n_faces: int = 6,
    n_levels: int = 19,
    level_embed_dim: int = 32,
    face_embed_dim: int = 8,
    halo_encoder_hidden: int = 256,
    halo_encoder_out: int = 128,
    trunk_dims: tuple = (1024, 512, 256),
    dropout: float = 0.1,
    single_task: bool = False,
) -> None:
```

After the existing trunk setup (line 167), replace the output heads block (lines 169-174) with:

```python
    # --- Output heads ---
    final_dim = trunk_dims[-1]  # 256
    self.single_task = single_task
    self.head_rho_s = nn.Linear(final_dim, k_cubed)
    if not single_task:
        self.head_log_dnorm = nn.Linear(final_dim, 1)
        nn.init.constant_(self.head_log_dnorm.bias, -27.5)
        self.head_refine = nn.Linear(final_dim, 1)
```

Replace the return block in `forward()` (lines 215-221) with:

```python
    # Output heads — rho_s uses residual from rho0_s so the model only
    # needs to learn the small correction (rho_s - rho0_s).
    rho_s = self.head_rho_s(x) + rho0_s  # [B, 512]
    if self.single_task:
        return rho_s, None, None
    log_dnorm = self.head_log_dnorm(x).squeeze(-1)  # [B]
    refine_logit = self.head_refine(x).squeeze(-1)  # [B]
    return rho_s, log_dnorm, refine_logit
```

Update `build_model()` (line 224-237) to pass `single_task`:

```python
def build_model(cfg: dict) -> MRANet:
    """Construct MRANet from config dict."""
    m = cfg["model"]
    return MRANet(
        k_cubed=m["k_cubed"],
        n_faces=m["n_faces"],
        n_levels=m["n_levels"],
        level_embed_dim=m["level_embed_dim"],
        face_embed_dim=m["face_embed_dim"],
        halo_encoder_hidden=m["halo_encoder_hidden"],
        halo_encoder_out=m["halo_encoder_out"],
        trunk_dims=tuple(m["trunk_dims"]),
        dropout=m["dropout"],
        single_task=m.get("single_task", False),
    )
```

- [ ] **Step 4: Run model tests to verify they pass**

Run: `cd /gpfs/projects/rjh/ruhin/madness-ruhin && python -m pytest mra_nn/tests/test_model.py -v`

Expected: ALL PASS (both existing multi-task tests and new single-task tests)

- [ ] **Step 5: Write failing tests for SingleTaskLoss**

Add to `mra_nn/tests/test_losses.py`:

```python
from mra_nn.losses import SingleTaskLoss


def test_single_task_loss_output():
    stl = SingleTaskLoss(pos_rho_weight=10.0)
    B = 16
    batch = {
        "rho_s": torch.randn(B, 512),
        "negative": torch.zeros(B),
    }
    pred_rs = torch.randn(B, 512)
    total, components = stl(batch, pred_rs)
    assert "loss_rho_s" in components
    assert total.requires_grad


def test_single_task_loss_no_learnable_params():
    stl = SingleTaskLoss(pos_rho_weight=10.0)
    params = list(stl.parameters())
    assert len(params) == 0


def test_single_task_loss_pos_weight():
    """Positive samples should contribute more to the loss than negatives."""
    stl = SingleTaskLoss(pos_rho_weight=10.0)
    B = 16
    pred_rs = torch.randn(B, 512)
    target = torch.randn(B, 512)

    # All positive
    batch_pos = {"rho_s": target, "negative": torch.zeros(B)}
    loss_pos, _ = stl(batch_pos, pred_rs)

    # All negative
    batch_neg = {"rho_s": target, "negative": torch.ones(B)}
    loss_neg, _ = stl(batch_neg, pred_rs)

    # Same errors but positive-weighted loss should be higher
    # because weight normalizes differently
    assert loss_pos.item() != loss_neg.item()
```

- [ ] **Step 6: Run loss tests to verify they fail**

Run: `cd /gpfs/projects/rjh/ruhin/madness-ruhin && python -m pytest mra_nn/tests/test_losses.py -v -k "single_task"`

Expected: FAIL — `ImportError: cannot import name 'SingleTaskLoss'`

- [ ] **Step 7: Implement SingleTaskLoss**

Add to `mra_nn/losses.py` after the `UncertaintyWeightedLoss` class (after line 134):

```python
class SingleTaskLoss(nn.Module):
    """Weighted MSE loss for single-task rho_s training.

    No learnable parameters. Positive (in-tree) samples weighted higher
    to counteract the 87% negative imbalance in the dataset.
    """

    def __init__(self, pos_rho_weight: float = 10.0) -> None:
        super().__init__()
        self.pos_rho_weight = pos_rho_weight

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        pred_rho_s: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Parameters
        ----------
        batch : dict with keys rho_s, negative
        pred_rho_s : [B, 512]

        Returns
        -------
        total_loss : scalar
        components : dict with loss_rho_s (detached)
        """
        is_pos = (batch["negative"] == 0).float()
        sample_w = 1.0 + (self.pos_rho_weight - 1.0) * is_pos
        per_sample_mse = F.mse_loss(
            pred_rho_s, batch["rho_s"], reduction="none"
        ).mean(dim=-1)
        loss = (sample_w * per_sample_mse).sum() / sample_w.sum()
        return loss, {"loss_rho_s": loss.detach()}
```

- [ ] **Step 8: Run all loss tests to verify they pass**

Run: `cd /gpfs/projects/rjh/ruhin/madness-ruhin && python -m pytest mra_nn/tests/test_losses.py -v`

Expected: ALL PASS

- [ ] **Step 9: Commit model and loss changes**

```bash
cd /gpfs/projects/rjh/ruhin/madness-ruhin
git add mra_nn/model.py mra_nn/losses.py mra_nn/tests/test_model.py mra_nn/tests/test_losses.py
git commit -m "feat(mra-nn): add single_task mode for rho_s-only ablation"
```

---

### Task 2: Update training loop and config for single-task mode

**Files:**
- Modify: `mra_nn/train.py:1-321`
- Create: `mra_nn/configs/single_task.yaml`

**Interfaces:**
- Consumes: `MRANet(single_task=True)` returning `(rho_s, None, None)`
- Consumes: `SingleTaskLoss(pos_rho_weight)` with `forward(batch, pred_rho_s)` returning `(loss, components)`
- Produces: training run with single-task CSV metrics, checkpoints compatible with `diagnose.py`

- [ ] **Step 1: Create single_task.yaml config**

Create `mra_nn/configs/single_task.yaml`:

```yaml
# MRA-NN single-task ablation config
# Tests whether multi-task interference prevents rho_s learning.
# See: docs/superpowers/specs/2026-07-31-mra-nn-single-task-ablation-design.md

data:
  dataset_path: /gpfs/projects/rjh/ruhin/mra_nn/training_dataset.h5
  raw_data_dir: /gpfs/projects/rjh/ruhin/mra_nn/training_data
  train_molecules:
    - h2o
    - nh3
    - ch4
    - co2
    - hf
    - n2
    - co
    - hcn
    - c2h4
    - c2h6
    - h2co
    - hcl
    - ch3oh
  val_molecules:
    - ch3f
  test_molecules:
    - h2o2
    - c2h2

model:
  k: 8
  ndim: 3
  k_cubed: 512
  n_faces: 6
  n_levels: 19
  level_embed_dim: 32
  face_embed_dim: 8
  halo_encoder_hidden: 256
  halo_encoder_out: 128
  trunk_dims: [1024, 512, 256]
  dropout: 0.1
  single_task: true

training:
  batch_size: 4096
  max_epochs: 120
  lr: 2.0e-4
  min_lr: 1.0e-6
  weight_decay: 1.0e-4
  warmup_epochs: 5
  patience: 20
  num_workers: 4
  seed: 42

loss:
  pos_rho_weight: 10.0
  refine_pos_weight: 10.0

checkpoint:
  dir: /gpfs/projects/rjh/ruhin/mra_nn/checkpoints
```

- [ ] **Step 2: Modify train.py to support single-task mode**

The changes to `train.py` are in four places:

**2a. Import SingleTaskLoss (line 20):**

Change:
```python
from mra_nn.losses import UncertaintyWeightedLoss
```
To:
```python
from mra_nn.losses import SingleTaskLoss, UncertaintyWeightedLoss
```

**2b. Replace `train_one_epoch` function (lines 37-79):**

Replace the entire function with a version that handles both modes:

```python
def train_one_epoch(
    model: torch.nn.Module,
    loss_fn,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    single_task: bool = False,
) -> dict:
    """Train for one epoch. Returns dict of mean losses."""
    model.train()
    accum = {}
    n_batches = 0

    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            rs, ld, ref = model(
                batch["rho0_s"], batch["vnuc_s"],
                batch["halo_rho0"], batch["halo_vnuc"],
                batch["level"],
            )
            if single_task:
                total_loss, components = loss_fn(batch, rs)
            else:
                total_loss, components = loss_fn(batch, rs, ld, ref)

        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if not single_task:
            # Clamp log_sigma_rs so sigma_rho_s stays <= 1.
            with torch.no_grad():
                loss_fn.log_sigma_rs.clamp_(max=0.0)

        # Accumulate metrics
        for k, v in components.items():
            accum[k] = accum.get(k, 0.0) + v.item()
        accum["total_loss"] = accum.get("total_loss", 0.0) + total_loss.item()
        n_batches += 1

    return {k: v / n_batches for k, v in accum.items()}
```

**2c. Replace `evaluate` function (lines 82-138):**

Replace the entire function:

```python
@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loss_fn,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    single_task: bool = False,
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
            rs, ld, ref = model(
                batch["rho0_s"], batch["vnuc_s"],
                batch["halo_rho0"], batch["halo_vnuc"],
                batch["level"],
            )
            if single_task:
                total_loss, components = loss_fn(batch, rs)
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
```

**2d. Modify `main()` function — loss, optimizer, CSV, print, and training loop (lines 141-321):**

After `build_model` (line 176), replace the loss/optimizer/CSV/loop setup. The key changes:

Replace loss construction (lines 181-186):
```python
    # Loss
    loss_cfg = cfg["loss"]
    single_task = cfg["model"].get("single_task", False)
    if single_task:
        loss_fn = SingleTaskLoss(
            pos_rho_weight=loss_cfg.get("pos_rho_weight", 10.0),
        ).to(device)
    else:
        loss_fn = UncertaintyWeightedLoss(
            focal_gamma=loss_cfg["focal_gamma"],
            focal_alpha=loss_cfg["focal_alpha"],
            pos_rho_weight=loss_cfg.get("pos_rho_weight", 10.0),
        ).to(device)
```

Replace optimizer construction (lines 189-194) — SingleTaskLoss has no parameters:
```python
    # Optimizer (includes loss_fn's learnable sigmas for multi-task)
    train_cfg = cfg["training"]
    params = list(model.parameters()) + list(loss_fn.parameters())
    optimizer = torch.optim.AdamW(
        params,
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
    )
```

Replace CSV fields (lines 224-235):
```python
    # CSV logger
    csv_path = ckpt_dir / "metrics.csv"
    if single_task:
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
```

Replace print header (lines 241-243):
```python
    print(f"\nTraining for up to {max_epochs} epochs (patience={train_cfg['patience']})...")
    if single_task:
        print(f"{'Epoch':>5} {'LR':>10} {'TrainLoss':>10} {'PosValMSE':>11} {'Best':>5}")
        print("-" * 50)
    else:
        print(f"{'Epoch':>5} {'LR':>10} {'TrainLoss':>10} {'PosValMSE':>11} {'ValRefF1':>9} {'SigRs':>8} {'Best':>5}")
        print("-" * 67)
```

Replace train/eval calls in the loop (lines 248-251):
```python
        train_metrics = train_one_epoch(
            model, loss_fn, train_dl, optimizer, scaler, device,
            single_task=single_task,
        )
        val_metrics = evaluate(model, loss_fn, val_dl, device,
                               single_task=single_task)
```

Replace the per-epoch print (lines 295-300):
```python
        best_marker = "*" if is_best else ""
        if single_task:
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
```

Remove `loss_fn_state_dict` from checkpoint saves when single_task (no learnable params). Replace both save blocks (lines 271-292):
```python
        if is_best:
            best_val_rs_mse = val_metrics["pos_rho_s_mse"]
            patience_counter = 0
        else:
            patience_counter += 1

        ckpt_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_rs_mse": best_val_rs_mse,
            "config": cfg,
        }
        if not single_task:
            ckpt_data["loss_fn_state_dict"] = loss_fn.state_dict()

        if is_best:
            torch.save(ckpt_data, ckpt_dir / "best.pt")

        # Always save last
        torch.save(ckpt_data, ckpt_dir / "last.pt")
```

- [ ] **Step 3: Verify all existing tests still pass**

Run: `cd /gpfs/projects/rjh/ruhin/madness-ruhin && python -m pytest mra_nn/tests/ -v`

Expected: ALL PASS — multi-task behavior unchanged

- [ ] **Step 4: Dry-run single-task config loads correctly**

Run: `cd /gpfs/projects/rjh/ruhin/madness-ruhin && python -c "
import yaml
from mra_nn.model import build_model
from mra_nn.losses import SingleTaskLoss
cfg = yaml.safe_load(open('mra_nn/configs/single_task.yaml'))
model = build_model(cfg)
print(f'single_task: {model.single_task}')
print(f'has head_log_dnorm: {hasattr(model, \"head_log_dnorm\")}')
print(f'params: {sum(p.numel() for p in model.parameters()):,}')
loss_fn = SingleTaskLoss(pos_rho_weight=cfg[\"loss\"][\"pos_rho_weight\"])
print(f'loss params: {len(list(loss_fn.parameters()))}')
"`

Expected output:
```
single_task: True
has head_log_dnorm: False
params: 3,043,600
loss params: 0
```

- [ ] **Step 5: Commit training loop and config changes**

```bash
cd /gpfs/projects/rjh/ruhin/madness-ruhin
git add mra_nn/train.py mra_nn/configs/single_task.yaml
git commit -m "feat(mra-nn): single-task training loop and config for ablation"
```

---

### Task 3: Submit single-task training job and run diagnostic

**Files:**
- Modify: `mra_nn/slurm/train_a100.sh` (or create a copy)
- Existing: `mra_nn/diagnose.py` (no changes needed)

**Interfaces:**
- Consumes: `mra_nn/configs/single_task.yaml`, `mra_nn/train.py` with single-task support
- Produces: checkpoint at `/gpfs/projects/rjh/ruhin/mra_nn/checkpoints/<run_name>/best.pt`, diagnostic output comparing train/val pos MSE against baselines

- [ ] **Step 1: Read existing Slurm script**

Read `mra_nn/slurm/train_a100.sh` to confirm the current structure.

- [ ] **Step 2: Submit training job with single-task config**

Submit a training job pointing to the single-task config. Either modify the existing script or run directly:

```bash
cat > /gpfs/projects/rjh/ruhin/mra_nn/slurm/train_single_task.sh << 'SLURM'
#!/bin/bash
#SBATCH --partition=a100-long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --job-name=mra-nn-st
#SBATCH --output=/gpfs/projects/rjh/ruhin/mra_nn/logs/train_st_%j.out
#SBATCH --error=/gpfs/projects/rjh/ruhin/mra_nn/logs/train_st_%j.err

export PATH="/cm/shared/apps/slurm/21.08.8/bin:$PATH"
source /gpfs/projects/rjh/ruhin/mra_nn/.venv/bin/activate
export PYTHONPATH=/gpfs/projects/rjh/adrian/pymra/src:$PYTHONPATH

python /gpfs/projects/rjh/ruhin/madness-ruhin/mra_nn/train.py \
    --config /gpfs/projects/rjh/ruhin/madness-ruhin/mra_nn/configs/single_task.yaml
SLURM

/cm/shared/apps/slurm/21.08.8/bin/sbatch /gpfs/projects/rjh/ruhin/mra_nn/slurm/train_single_task.sh
```

Monitor with: `/cm/shared/apps/slurm/21.08.8/bin/squeue -u ruhipatel`

Expected: Job runs ~2 hours on A100, produces checkpoints at `/gpfs/projects/rjh/ruhin/mra_nn/checkpoints/<run_name>/`

- [ ] **Step 3: Update diagnose.py checkpoint path and run diagnostic**

After training completes, update the `ckpt_dir` in `diagnose.py` to point to the new run's checkpoint, or pass it as a CLI argument. Then submit the diagnostic:

```bash
# Update ckpt_dir in diagnose.py to the new run_name, then:
/cm/shared/apps/slurm/21.08.8/bin/sbatch /gpfs/projects/rjh/ruhin/mra_nn/slurm_diagnose.sh
```

Expected output (one of three outcomes):

**Outcome A (multi-task confirmed):**
```
Train          3.811e-07       X.XXXe-08    0.XXx      YES
Val            4.846e-07       X.XXXe-07    X.XXx       NO
```

**Outcome B (multi-task was sole bottleneck):**
```
Train          3.811e-07       X.XXXe-08    0.XXx      YES
Val            4.846e-07       X.XXXe-07    0.XXx      YES
```

**Outcome C (architecture insufficient):**
```
Train          3.811e-07       X.XXXe-07    X.XXx       NO
Val            4.846e-07       X.XXXe-06    X.XXx       NO
```

- [x] **Step 4: Record result and determine next step**

### Results (2026-07-31)

**Training job:** Slurm 2107567, completed 120 epochs on A100.
**Checkpoint:** `/gpfs/projects/rjh/ruhin/mra_nn/checkpoints/2026-07-31_04-45/best.pt`

**Outcome: C (architecture insufficient)** — single-task mode improved significantly over multi-task but still does not beat baseline.

| Split | Baseline (rho0) | Model MSE | Ratio | Beats? |
|-------|-----------------|-----------|-------|--------|
| Train | 3.811e-07       | 4.867e-07 | 1.28x | NO     |
| Val   | 4.846e-07       | 4.867e-07 | 1.00x | NO     |

**Improvement from multi-task:** Train went from 2.99x → 1.28x, val from 2.84x → 1.00x. Multi-task interference was a major factor but not the only one.

### Per-Level Head Diagnostic (`diagnose_head.py`)

- mean |head(x)| = 5.1e-5 (0.66x of mean |Δ| = 7.7e-5) — head is outputting corrections, not zero
- head RMS: 1.6e-4 vs Δ RMS: 1.9e-4
- **Levels 11-12 beat baseline** (~0.99x) — correction is learnable at levels with many samples
- **Levels 0-7 and 15-17 hurt** — model adds noise at extreme levels with few training samples

### Level Masking Simulation (`estimate_masked.py`)

Even zeroing head output at under-represented levels:

| min_count | Masked levels | Model ratio |
|-----------|---------------|-------------|
| 50        | [0,1,15,16,17]| 1.006x      |
| 100       | [0,1,2,15,16,17]| 1.006x   |
| 200       | [0,1,2,3,15,16,17]| 1.008x |
| 500       | [0..5,14..17] | 1.067x      |

Best achievable with masking: 1.006x — still 0.6% worse than baseline.

### Conclusion

The single-task ablation confirmed multi-task interference was a major bottleneck (2.99x → 1.28x train). However, even without multi-task heads, the model cannot beat the rho0 baseline. Per-level analysis shows the correction IS learnable at well-represented levels (11-12), but noise at extreme levels dominates the aggregate.

---

## Context for Next Session

### What was tried and ruled out (do not re-propose)

1. **"Just add more molecules" alone** — the model can't fit its *existing* training data (train ratio 1.28x, no train/val gap). More data helps generalization, but that's not the problem here. However, more molecules *combined* with level masking (Option A) is viable — it addresses data-starvation at extreme levels, which is a different problem.

2. **Near-zero head initialization** — the training curve shows the model passed through the baseline neighborhood (epochs 40-55, val MSE 7.9e-7 → 4.9e-7, baseline 4.8e-7 for val) and plateaued above it. The optimization found this region and couldn't break through — init wouldn't help.

3. **Level masking at inference only** — tested exhaustively (see masking simulation table above). Best case 1.006x, still above baseline.

### Training curve shape

- Epochs 0-39: rapid improvement, val pos MSE dropped 2.5e-2 → 7.9e-7 (5 orders of magnitude)
- Epochs 40-55: approached baseline, val MSE 7.9e-7 → 5.0e-7
- Epochs 55-120: asymptotic plateau at 4.87e-7, never crossing below val baseline (4.846e-7)
- This is a capacity/information ceiling, not a training failure

Full training log: `/gpfs/projects/rjh/ruhin/mra_nn/logs/train_st_2107567.out`

### Per-level sample counts (from `diagnose_head.py`)

These took a dedicated diagnostic job — don't re-derive. Key pattern: performance correlates with sample count.

| Level | Count  | Ratio  | Beats? |
|-------|--------|--------|--------|
| 3     | ~30    | >1x    | no     |
| 4-7   | ~100-500 | >1x  | no     |
| 8     | ~2k    | ~1.0x  | borderline |
| 9     | ~8k    | ~1.0x  | borderline |
| 10    | ~20k   | ~1.0x  | borderline |
| 11    | ~24k   | ~0.99x | YES    |
| 12    | ~13k   | ~0.99x | YES    |
| 13    | ~5k    | ~1.0x  | borderline |
| 14-17 | <1k    | >1x    | no     |

### Open Options

1. **Option A: More training data + level masking during training**
   - **Why:** The model learns corrections at levels 11-12 (thousands of samples) but adds noise at levels 0-7 and 15-17 (few samples). More molecules (W4-11 set) increase sample counts at all levels; level-aware weighting/masking during training prevents under-represented levels from polluting gradients.
   - **Evidence:** See per-level table above — levels 11-12 (24k+13k samples) achieve ~0.99x; levels 0-3 (<100 samples) consistently hurt.
   - **Effort:** ~1-2 sessions. Run `dump_training_functions` for W4-11, rebuild HDF5. Level masking is a ~10-line change to `SingleTaskLoss`.

2. **Option B: Richer context (parent node features)**
   - **Why:** The correction Δ encodes how SCF redistributes density across the tree. The model sees only 6 face-adjacent halo neighbors at the *same* level — no coarser/finer scale information. In MRA, parent coefficients carry low-frequency structure that determines density flow between levels. The head outputs corrections of the right magnitude (0.66x of |Δ|) but in the wrong direction at many levels — suggestive of missing context rather than capacity.
   - **Effort:** ~3-4 sessions. Requires modifying `dump_training_functions` (C++ in MADNESS), updating `dataset_builder.py`, and modifying the model encoder.

3. **Option C: Skip Gate 1 and test SCF impact directly**
   - **Why:** Gate 1 (aggregate MSE < baseline) may be too strict. The model beats baseline at levels 11-12, which contain the bulk of electron density (~37k of ~175k positive samples). SCF may converge faster even if aggregate MSE is worse, because corrections at the *important* levels are better.
   - **Effort:** ~1 session. Use existing `predict.py` to generate initial guess, run `moldft` with it, count iterations vs rho0. **Prerequisite:** verify moldft supports custom initial density input (check `moldft --help` or MADNESS source).

### Relationships between options

- **A and C are complementary** — C tests whether the current model is already useful for SCF (cheap, fast feedback); A improves the model. Pursue C first, then A if C shows promise.
- **B is independent** — higher effort, higher potential payoff. Can be pursued in parallel with A.
- **A alone may not be sufficient** — the model might still plateau near baseline without richer context (B). But A is the cheapest experiment.

### Uncommitted diagnostic scripts (on disk, not in git)

At `/gpfs/projects/rjh/ruhin/madness-ruhin/mra_nn/`:
- `diagnose.py` — loads checkpoint, computes train/val pos MSE vs baseline, prints diagnosis
- `diagnose_head.py` — per-level head output norms and MSE breakdown
- `estimate_masked.py` — simulates level masking at various thresholds

At `/gpfs/projects/rjh/ruhin/mra_nn/`:
- `slurm_diagnose.sh` — submits diagnostic scripts to `debug-40core` partition (CPU, 30min)
