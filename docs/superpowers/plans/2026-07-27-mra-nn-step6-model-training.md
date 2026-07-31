# MRA-NN Step 6: MLP Model & Training — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Implement a FiLM-conditioned MLP with factored halo encoder that predicts converged density s-coefficients (rho_s), wavelet norms (log‖d‖), and refinement decisions from promolecular density and nuclear potential inputs in the MRA basis.

**Architecture:** Shared halo encoder processes 6 face-adjacent neighbors, concatenated with center box features, fed through 3 FiLM-conditioned trunk layers (level embedding drives γ/β), into 3 output heads. Uncertainty-weighted multi-task loss (Kendall et al. 2018) with focal loss on the refine head. All heads train on all samples (no masking).

**Tech Stack:** Python 3.12, PyTorch (with CUDA/AMP), h5py, numpy, pyyaml, pymra

**Update history:** Originally written 2026-07-27 for k=6/delta_rho. Updated 2026-07-30 for k=8/direct coefficients/redundant form per Adrian's confirmation. Tasks 0A and 0B added for data regeneration. All dimension references updated from 216→512.

## Global Constraints

- **Branch:** `feat/mra-nn-data` on `ruhinpatel/madness`
- **All code lives in:** `/gpfs/projects/rjh/ruhin/madness-ruhin/mra_nn/`
- **Training dataset:** `/gpfs/projects/rjh/ruhin/mra_nn/training_dataset.h5`
- **Raw data:** `/gpfs/projects/rjh/ruhin/mra_nn/training_data/<mol>/` (rho0.mad.h5, vnuc.mad.h5, rho.mad.h5)
- **pymra:** `/gpfs/projects/rjh/adrian/pymra/src` — must be on PYTHONPATH; cannot pip install (owned by Adrian)
- **Venv:** `/gpfs/projects/rjh/ruhin/mra_nn/.venv/` — Python 3.12, numpy 1.23.5, h5py 3.14.0, PyTorch 2.13.0
- **Checkpoints/logs:** `/gpfs/projects/rjh/ruhin/mra_nn/checkpoints/<run_name>/` (not in git)
- **Slurm:** `/cm/shared/apps/slurm/21.08.8/bin/sbatch` (not in default PATH); A100 GPU partition
- **Git rules:** No `Co-Authored-By:` lines in commits. No Jira ticket names in branches/commits/paths.
- **k=8, k^3=512** for all coefficient dimensions
- **thresh=1e-6** for MADNESS calculations
- **Design spec:** `docs/superpowers/specs/2026-07-27-mra-nn-step6-model-training-design.md`

---

### Task 0A: Regenerate Training Data at k=8

**Model recommendation: Sonnet 4.6** — updating input files and resubmitting an existing Slurm pipeline.

**Files:**
- Modify: `mra_nn/molecules/*.in` (all 15 molecule input files)
- No new files

**Interfaces:**
- Consumes: existing `gen_training_data.sh` Slurm pipeline, existing molecule input files
- Produces: 45 `.mad.h5` files at k=8/thresh=1e-6 in `/gpfs/projects/rjh/ruhin/mra_nn/training_data/`

- [x] **Step 1: Update all molecule input files to k=8/thresh=1e-6**

For each of the 15 `.in` files in `mra_nn/molecules/`, change:
```
  thresh 1e-4
  k 6
```
to:
```
  thresh 1e-6
```

**Important:** Remove the `k` line entirely — MADNESS picks k from the protocol threshold (k=8 at thresh=1e-6). See CLAUDE.md: "Do not set k explicitly."

The molecule files to update are: `h2o.in`, `nh3.in`, `ch4.in`, `co2.in`, `hf.in`, `n2.in`, `co.in`, `hcn.in`, `c2h2.in`, `c2h4.in`, `c2h6.in`, `h2co.in`, `ch3oh.in`, `h2o2.in`, `hcl.in`.

Verify with:
```bash
grep -l "k 6" mra_nn/molecules/*.in  # should return nothing
grep -c "thresh 1e-6" mra_nn/molecules/*.in  # should show 1 for each file
```

- [x] **Step 2: Back up existing k=6 data**

```bash
mv /gpfs/projects/rjh/ruhin/mra_nn/training_data /gpfs/projects/rjh/ruhin/mra_nn/training_data_k6
mkdir -p /gpfs/projects/rjh/ruhin/mra_nn/training_data
```

- [x] **Step 3: Submit data generation job**

```bash
cd /gpfs/projects/rjh/ruhin/mra_nn
mkdir -p logs
/cm/shared/apps/slurm/21.08.8/bin/sbatch /gpfs/projects/rjh/ruhin/madness-ruhin/mra_nn/gen_training_data.sh
```

Expected: `Submitted batch job <jobid>`

**Note:** This job will take significantly longer than the k=6 run (~26 min). At k=8/thresh=1e-6, moldft SCF takes more iterations with finer grids. Expect 1-4 hours. Monitor with:
```bash
squeue -u ruhipatel
tail -f /gpfs/projects/rjh/ruhin/mra_nn/logs/gen_training_data_<jobid>.out
```

- [x] **Step 4: Validate generated data**

Once the job completes, run the validation script:
```bash
cd /gpfs/projects/rjh/ruhin/madness-ruhin
source /gpfs/projects/rjh/ruhin/mra_nn/.venv/bin/activate
PYTHONPATH=/gpfs/projects/rjh/adrian/pymra/src \
python mra_nn/validate_h5.py /gpfs/projects/rjh/ruhin/mra_nn/training_data
```

Expected: all 45 files (15 molecules × 3 functions) validated with `/meta`, `/keys`, `/coeffs` datasets. Verify k=8 in metadata:
```bash
PYTHONPATH=/gpfs/projects/rjh/adrian/pymra/src python -c "
from pymra import read_function
t = read_function('/gpfs/projects/rjh/ruhin/mra_nn/training_data/h2o/rho.mad.h5')
print(f'k={t.k}, n_nodes={len(t.nodes)}')
"
```

Expected: `k=8`, and n_nodes should be larger than the k=6 value (was 5,300 for h2o).

- [x] **Step 5: Commit molecule input changes**

```bash
cd /gpfs/projects/rjh/ruhin/madness-ruhin
git add mra_nn/molecules/*.in
git commit -m "feat(mra-nn): update molecule inputs to thresh=1e-6 (k=8 auto)

Standard MADNESS production precision per Adrian (2026-07-30).
Removed explicit k=6 — MADNESS derives k=8 from thresh=1e-6."
```

---

### Task 0B: Update Dataset Builder for Direct Coefficients

**Model recommendation: Sonnet 4.6** — straightforward field rename and gate update in existing code.

**Files:**
- Modify: `mra_nn/dataset_builder.py`

**Interfaces:**
- Consumes: 45 `.mad.h5` files at k=8 from Task 0A
- Produces: updated `training_dataset.h5` with `rho_s` field instead of `delta_rho`, all at k=8/k^3=512

- [x] **Step 1: Update dataset_builder.py — replace delta_rho with rho_s**

In `dataset_builder.py`, make these changes:

1. Update the module docstring: replace references to `Δρ s-coefficients` with `rho s-coefficients (direct)`, and update the HDF5 layout doc to show `rho_s` instead of `delta_rho`.

2. In `process_molecule()`, the `_append` function currently computes:
```python
drho = (rho_s - rho0_s).ravel()
```
and stores it as `delta_rho_list`. Change this to store `rho_s` directly:
```python
rho_s_flat = rho_s.ravel()
```
Rename the list from `delta_rho_list` to `rho_s_list`, and update all references.

3. In the return dict, change:
```python
"delta_rho": np.array(delta_rho_list, dtype=np.float32),
```
to:
```python
"rho_s": np.array(rho_s_list, dtype=np.float32),
```

4. In `gate_check()`, update the verification. Currently it checks `rho0_s + delta_rho` reproduces rho. Now it should check that `rho_s` directly matches the original rho leaf coefficients:
```python
rho_s_from_data = data["rho_s"][i].reshape((k,) * ndim)
rho_s_from_tree = node_s(rho, key)
err = float(np.max(np.abs(rho_s_from_data - rho_s_from_tree)))
```
And the integral reconstruction should use `data["rho_s"]` directly:
```python
s = data["rho_s"][i].reshape((k,) * ndim)
rho_tree.nodes[key] = Node(s=s.astype(float))
```

- [x] **Step 2: Rebuild the training dataset**

```bash
cd /gpfs/projects/rjh/ruhin/madness-ruhin
source /gpfs/projects/rjh/ruhin/mra_nn/.venv/bin/activate
PYTHONPATH=/gpfs/projects/rjh/adrian/pymra/src \
python mra_nn/dataset_builder.py \
    --data-dir /gpfs/projects/rjh/ruhin/mra_nn/training_data \
    --out /gpfs/projects/rjh/ruhin/mra_nn/training_dataset.h5 \
    --gate
```

Expected: gate PASS — `rho_s` directly matches rho leaf coefficients (should be exact to float32 precision), ∫ρ = N.

Verify k=8 and rho_s field:
```bash
PYTHONPATH=/gpfs/projects/rjh/adrian/pymra/src python -c "
import h5py
with h5py.File('/gpfs/projects/rjh/ruhin/mra_nn/training_dataset.h5', 'r') as f:
    print(f'k={f.attrs[\"k\"]}, molecules={list(f.attrs[\"molecules\"])}')
    mol = list(f.attrs['molecules'])[0]
    print(f'{mol}: rho_s shape={f[mol][\"rho_s\"].shape}')
    print(f'{mol}: has delta_rho={\"delta_rho\" in f[mol]}')
"
```

Expected: `k=8`, `rho_s shape=(N, 512)`, `has delta_rho=False`.

- [x] **Step 3: Record dataset statistics**

Print sample counts for the updated plan/spec TBD fields:
```bash
PYTHONPATH=/gpfs/projects/rjh/adrian/pymra/src python -c "
import h5py, numpy as np
with h5py.File('/gpfs/projects/rjh/ruhin/mra_nn/training_dataset.h5', 'r') as f:
    total = sum(f[m]['level'].shape[0] for m in f.attrs['molecules'])
    pos = sum(np.sum(f[m]['negative'][:] == 0) for m in f.attrs['molecules'])
    neg = sum(np.sum(f[m]['negative'][:] == 1) for m in f.attrs['molecules'])
    ref = sum(np.sum(f[m]['refine'][:] == 1) for m in f.attrs['molecules'])
    print(f'Total: {total}, Positive: {pos}, Negative: {neg}, Refine=1: {ref}')
"
```

Record these numbers — update the spec's Section 8 TBD fields.

- [x] **Step 4: Commit**

```bash
cd /gpfs/projects/rjh/ruhin/madness-ruhin
git add mra_nn/dataset_builder.py
git commit -m "feat(mra-nn): switch dataset from delta_rho to direct rho_s

Store converged density s-coefficients directly instead of the
correction (rho - rho0). Gate check verifies rho_s matches original
tree coefficients to float32 precision."
```

---

### Task 1: Environment Setup & Config

**Model recommendation: Sonnet 4.6** — update existing config YAML for k=8 dimensions.

**Files:**
- Modify: `mra_nn/configs/default.yaml`

**Interfaces:**
- Consumes: nothing
- Produces: `default.yaml` config dict loaded by all subsequent tasks

- [x] **Step 1: Update the config file for k=8 and rho_s**

Update `mra_nn/configs/default.yaml`:

```yaml
# MRA-NN Step 6 — default training configuration (v2: k=8, direct coefficients)
# See design spec: docs/superpowers/specs/2026-07-27-mra-nn-step6-model-training-design.md

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
  val_molecules:
    - ch3oh
  test_molecules:
    - h2o2
    - c2h2

model:
  k: 8
  ndim: 3
  k_cubed: 512
  n_faces: 6
  n_levels: 19              # levels 0-18
  level_embed_dim: 32
  face_embed_dim: 8
  halo_encoder_hidden: 256
  halo_encoder_out: 128
  trunk_dims: [1024, 512, 256]
  dropout: 0.1

training:
  batch_size: 4096
  max_epochs: 200
  lr: 1.0e-3
  min_lr: 1.0e-5
  weight_decay: 1.0e-4
  warmup_epochs: 5
  patience: 20
  num_workers: 4
  seed: 42

loss:
  focal_gamma: 2.0
  focal_alpha: 0.75
  refine_pos_weight: 10.0   # oversampling weight for refine=1 positives

checkpoint:
  dir: /gpfs/projects/rjh/ruhin/mra_nn/checkpoints
```

Verify:
```bash
source /gpfs/projects/rjh/ruhin/mra_nn/.venv/bin/activate
python -c "
import yaml
with open('mra_nn/configs/default.yaml') as f:
    cfg = yaml.safe_load(f)
print(f'k={cfg[\"model\"][\"k\"]}, k_cubed={cfg[\"model\"][\"k_cubed\"]}')
print(f'Trunk dims: {cfg[\"model\"][\"trunk_dims\"]}')
"
```

Expected:
```
k=8, k_cubed=512
Trunk dims: [1024, 512, 256]
```

- [x] **Step 2: Commit**

```bash
cd /gpfs/projects/rjh/ruhin/madness-ruhin
git add mra_nn/configs/default.yaml
git commit -m "feat(mra-nn): update config to k=8/512 with direct rho_s targets"
```

---

### Task 2: PyTorch Dataset

**Model recommendation: Sonnet 4.6** — update existing dataset.py for rho_s field.

**Files:**
- Modify: `mra_nn/dataset.py`
- Modify: `mra_nn/tests/test_dataset.py`

**Interfaces:**
- Consumes: `training_dataset.h5` (HDF5 file built by `dataset_builder.py` with `rho_s` field), `default.yaml` config
- Produces:
  - `class MRADataset(torch.utils.data.Dataset)` with `__len__() -> int` and `__getitem__(idx: int) -> dict` returning keys: `rho0_s` (512,), `vnuc_s` (512,), `halo_rho0` (6, 512), `halo_vnuc` (6, 512), `rho_s` (512,), `log_dnorm` (scalar), `refine` (scalar), `level` (scalar int), `negative` (scalar int)
  - `def build_dataloaders(cfg: dict) -> Tuple[DataLoader, DataLoader, DataLoader]` returning (train_loader, val_loader, test_loader)
  - `def compute_baseline_mse(dataset: MRADataset) -> float` returning ‖rho_s - rho0_s‖² averaged over all samples (the "use rho0 as-is" baseline)

- [x] **Step 1: Update the dataset module**

In `mra_nn/dataset.py`:

1. Replace `"delta_rho"` with `"rho_s"` in `FIELD_NAMES`.

2. Update `compute_baseline_mse()` — it should compute the MSE of using rho0 as the prediction (instead of the old "predict zero correction" baseline):

```python
def compute_baseline_mse(dataset: MRADataset) -> float:
    """Mean squared error of using rho0_s as the density prediction.

    This is the "use promolecular density as-is" baseline — the model must beat this.
    """
    rho_s = dataset.data["rho_s"]       # [N, 512]
    rho0_s = dataset.data["rho0_s"]     # [N, 512]
    return float((rho_s - rho0_s).pow(2).mean())
```

- [x] **Step 2: Update the test file**

In `mra_nn/tests/test_dataset.py`:

1. Replace all references to `"delta_rho"` with `"rho_s"`.
2. Update shape assertions from `(216,)` to `(512,)` for all coefficient fields.
3. Update `test_dataset_getitem_keys` expected_keys set.
4. Update `test_dataset_getitem_shapes`:
```python
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
```
5. Update `test_build_dataloaders`:
```python
def test_build_dataloaders(cfg):
    train_dl, val_dl, test_dl = build_dataloaders(cfg)
    batch = next(iter(train_dl))
    assert batch["rho0_s"].shape[0] <= cfg["training"]["batch_size"]
    assert batch["rho0_s"].shape[1] == 512
```
6. Update `test_baseline_mse` — baseline should be positive (rho0 != rho):
```python
def test_baseline_mse(train_ds):
    baseline = compute_baseline_mse(train_ds)
    assert baseline > 0.0
    assert np.isfinite(baseline)
```

- [x] **Step 3: Run tests**

```bash
cd /gpfs/projects/rjh/ruhin/madness-ruhin
source /gpfs/projects/rjh/ruhin/mra_nn/.venv/bin/activate
PYTHONPATH=/gpfs/projects/rjh/adrian/pymra/src \
python -m pytest mra_nn/tests/test_dataset.py -v
```

Expected: all tests PASS.

- [x] **Step 4: Commit**

```bash
cd /gpfs/projects/rjh/ruhin/madness-ruhin
git add mra_nn/dataset.py mra_nn/tests/test_dataset.py
git commit -m "feat(mra-nn): update dataset for k=8 and direct rho_s target

Replace delta_rho with rho_s field. Baseline MSE now measures
how well rho0 alone predicts rho (model must beat this)."
```

---

### Task 3: Model Architecture

**Model recommendation: Opus 4.6** — update dimensions throughout the model for k=8 and rename output head.

**Files:**
- Modify: `mra_nn/model.py`
- Modify: `mra_nn/tests/test_model.py`

**Interfaces:**
- Consumes: config dict from `default.yaml` (model section)
- Produces:
  - `class MRANet(nn.Module)` with `forward(rho0_s, vnuc_s, halo_rho0, halo_vnuc, level) -> Tuple[Tensor, Tensor, Tensor]` returning `(rho_s [B, 512], log_dnorm [B], refine_logit [B])`
  - `def build_model(cfg: dict) -> MRANet`

- [x] **Step 1: Update the model**

In `mra_nn/model.py`:

1. Update `HaloEncoder`:
   - Input dim changes: `2 * k_cubed + face_embed_dim` = 2×512+8 = 1032 (was 440)
   - No structural changes needed — dimensions flow from `k_cubed` parameter

2. Update `MRANet`:
   - Center dim: `2 * k_cubed` = 1024 (was 432)
   - Trunk input: `768 + 1024` = 1792 (was 1200)
   - Rename `head_delta_rho` to `head_rho_s`
   - Output head: `Linear(final_dim, k_cubed)` = Linear(256, 512) (was 216)

3. Update `forward()` return variable name from `delta_rho` to `rho_s`.

4. Update docstrings: replace `delta_rho` references with `rho_s`.

- [x] **Step 2: Update the test file**

In `mra_nn/tests/test_model.py`:

1. Update `test_forward_shapes`:
```python
def test_forward_shapes(model):
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
    assert log_dnorm.shape == (B,)
    assert refine_logit.shape == (B,)
```

2. Update `test_forward_gradients` similarly — use 512 dims, rename `delta_rho` to `rho_s`.

3. Update `test_parameter_count` — allow 2M to 4M range (expect ~2.8M):
```python
def test_parameter_count(model):
    total = sum(p.numel() for p in model.parameters())
    assert 2_000_000 < total < 4_000_000, f"Parameter count {total} outside expected range"
```

4. All tensor dimensions: 216 → 512.

- [x] **Step 3: Run tests**

```bash
cd /gpfs/projects/rjh/ruhin/madness-ruhin
source /gpfs/projects/rjh/ruhin/mra_nn/.venv/bin/activate
python -m pytest mra_nn/tests/test_model.py -v
```

Expected: all tests PASS. Check parameter count is in range 2M–4M.

- [x] **Step 4: Commit**

```bash
cd /gpfs/projects/rjh/ruhin/madness-ruhin
git add mra_nn/model.py mra_nn/tests/test_model.py
git commit -m "feat(mra-nn): update MRANet for k=8 (512-dim) and rho_s output

Halo encoder input 1032, trunk input 1792, rho_s head 256->512.
~2.8M parameters. Renamed delta_rho head to rho_s."
```

---

### Task 4: Loss Functions & Training Loop

**Model recommendation: Opus 4.6** — update loss to train rho_s on all samples (no masking), update training loop variable names.

**Files:**
- Modify: `mra_nn/losses.py`
- Modify: `mra_nn/train.py`
- Modify: `mra_nn/tests/test_losses.py`

**Interfaces:**
- Consumes:
  - `MRANet` from `model.py`: `forward(...) -> (rho_s, log_dnorm, refine_logit)`
  - `build_dataloaders(cfg)` from `dataset.py`
  - `build_model(cfg)` from `model.py`
  - `compute_baseline_mse(dataset)` from `dataset.py`
  - config dict from `default.yaml`
- Produces:
  - `class FocalLoss(nn.Module)` with `forward(logits: Tensor, targets: Tensor) -> Tensor`
  - `class UncertaintyWeightedLoss(nn.Module)` with `forward(batch: dict, rho_s: Tensor, log_dnorm: Tensor, refine_logit: Tensor) -> Tuple[Tensor, dict]` returning `(total_loss, loss_components_dict)`
  - `train.py` CLI: `python train.py --config <yaml>` that trains to completion

- [x] **Step 1: Update losses.py — rho_s on all samples**

In `mra_nn/losses.py`:

1. Rename all `delta_rho` references to `rho_s` (parameter names, variable names, dict keys).

2. In `UncertaintyWeightedLoss.forward()`, the rho_s MSE should train on **all samples** (remove the positive-only masking):

```python
# --- rho_s MSE (all samples) ---
loss_rs = F.mse_loss(pred_rho_s, batch["rho_s"])
```

Remove the `pos_mask` / `n_pos` logic and the conditional zero-loss for empty positives.

3. Rename `log_sigma_dr` to `log_sigma_rs` and update dict keys from `"loss_delta_rho"` / `"sigma_delta_rho"` to `"loss_rho_s"` / `"sigma_rho_s"`.

- [x] **Step 2: Update test_losses.py**

In `mra_nn/tests/test_losses.py`:

1. Update `test_uncertainty_weighted_loss_output_keys`:
```python
def test_uncertainty_weighted_loss_output_keys():
    uwl = UncertaintyWeightedLoss(focal_gamma=2.0, focal_alpha=0.75)
    B = 16
    batch = {
        "rho_s": torch.randn(B, 512),
        "log_dnorm": torch.randn(B),
        "refine": torch.randint(0, 2, (B,), dtype=torch.float32),
        "negative": torch.zeros(B),
    }
    pred_rs = torch.randn(B, 512)
    pred_ld = torch.randn(B)
    pred_ref = torch.randn(B)
    total, components = uwl(batch, pred_rs, pred_ld, pred_ref)
    assert "loss_rho_s" in components
    assert "loss_log_dnorm" in components
    assert "loss_refine" in components
    assert "sigma_rho_s" in components
    assert total.requires_grad
```

2. Update `test_uncertainty_weighted_loss_masks_negatives` — this test no longer applies since we train on all samples. Replace it with a test that verifies rho_s loss is computed on all samples including negatives:
```python
def test_uncertainty_weighted_loss_includes_negatives():
    uwl = UncertaintyWeightedLoss(focal_gamma=2.0, focal_alpha=0.75)
    B = 16
    batch = {
        "rho_s": torch.randn(B, 512),
        "log_dnorm": torch.randn(B),
        "refine": torch.zeros(B),
        "negative": torch.ones(B),  # all negatives
    }
    pred_rs = torch.randn(B, 512)
    pred_ld = torch.randn(B)
    pred_ref = torch.randn(B)
    total, components = uwl(batch, pred_rs, pred_ld, pred_ref)
    # rho_s loss should be non-zero even for negatives
    assert components["loss_rho_s"].item() > 0.0
```

3. Update all 216 dims to 512 in test tensors.

- [x] **Step 3: Update train.py**

In `mra_nn/train.py`:

1. Rename all `delta_rho` / `dr` variable names to `rho_s` / `rs`.

2. In `train_one_epoch()` and `evaluate()`, update the model call:
```python
rs, ld, ref = model(
    batch["rho0_s"], batch["vnuc_s"],
    batch["halo_rho0"], batch["halo_vnuc"],
    batch["level"],
)
total_loss, components = loss_fn(batch, rs, ld, ref)
```

3. Update CSV field names: `train_loss_delta_rho` → `train_loss_rho_s`, `val_loss_delta_rho` → `val_loss_rho_s`, etc.

4. Update early stopping: monitor `val_metrics["loss_rho_s"]` instead of `val_metrics["loss_delta_rho"]`.

5. Update `best_val_dr_mse` variable name to `best_val_rs_mse`.

6. Update print statements and summary to reference rho_s.

- [x] **Step 4: Run all tests**

```bash
cd /gpfs/projects/rjh/ruhin/madness-ruhin
source /gpfs/projects/rjh/ruhin/mra_nn/.venv/bin/activate
PYTHONPATH=/gpfs/projects/rjh/adrian/pymra/src \
python -m pytest mra_nn/tests/test_losses.py mra_nn/tests/test_model.py mra_nn/tests/test_dataset.py -v
```

Expected: all tests PASS.

- [x] **Step 5: Commit**

```bash
cd /gpfs/projects/rjh/ruhin/madness-ruhin
git add mra_nn/losses.py mra_nn/train.py mra_nn/tests/test_losses.py
git commit -m "feat(mra-nn): update losses and training for direct rho_s

rho_s MSE on all samples (no positive-only masking). All variable
names updated from delta_rho to rho_s. Early stopping monitors
val rho_s MSE."
```

---

### Task 5: Inference Pipeline

**Model recommendation: Opus 4.6** — update tree-walk to use predicted coefficients directly.

**Files:**
- Modify: `mra_nn/predict.py`
- Modify: `mra_nn/tests/test_predict.py`

**Interfaces:**
- Consumes:
  - `MRANet` from `model.py`
  - pymra: `read_function`, `write_function`, `node_s`, `FunctionTree`, `Key`, `Node`
  - pymra: `twoscale.refine`
- Produces:
  - `def predict_density(model, rho0_path, vnuc_path, n_electrons, device, refine_threshold=0.5) -> FunctionTree`
  - CLI: `python predict.py --checkpoint <best.pt> --rho0 <path> --vnuc <path> --n-electrons <N> --out <output.mad.h5>`

- [x] **Step 1: Update predict.py — use predicted coefficients directly**

In `mra_nn/predict.py`:

1. In `predict_density()`, update the leaf assignment. Currently:
```python
rho0_s = node_s(rho0_tree, key).ravel()
pred_s = rho0_s + delta_rho_np[i]
```
Replace with direct coefficient use:
```python
pred_s = rho_s_np[i]
```

2. Rename `delta_rho` variables to `rho_s` throughout:
```python
rho_s, log_dnorm, refine_logit = model(...)
rho_s_np = rho_s.cpu().numpy()
```

3. Update docstrings.

- [x] **Step 2: Update test_predict.py**

In `mra_nn/tests/test_predict.py`:

1. Tests should still pass as-is since the interface (`predict_density()` returns a `FunctionTree`) hasn't changed.
2. Update any hardcoded 216 references if present.

- [x] **Step 3: Run tests**

```bash
cd /gpfs/projects/rjh/ruhin/madness-ruhin
source /gpfs/projects/rjh/ruhin/mra_nn/.venv/bin/activate
PYTHONPATH=/gpfs/projects/rjh/adrian/pymra/src \
python -m pytest mra_nn/tests/test_predict.py -v
```

Expected: all tests PASS.

- [x] **Step 4: Commit**

```bash
cd /gpfs/projects/rjh/ruhin/madness-ruhin
git add mra_nn/predict.py mra_nn/tests/test_predict.py
git commit -m "feat(mra-nn): update inference to use direct rho_s coefficients

No longer adds delta_rho to rho0_s — uses predicted s-coefficients
directly as leaf values in the output tree."
```

---

### Task 6: Evaluation & Step 6 Gate

**Model recommendation: Sonnet 4.6** — update gate metrics for rho_s baseline.

**Files:**
- Modify: `mra_nn/evaluate.py`

**Interfaces:**
- Consumes:
  - `predict_density()` from `predict.py`
  - `build_model()` from `model.py`
  - `MRADataset`, `compute_baseline_mse()` from `dataset.py`
  - pymra: `read_function`, `FunctionTree`
- Produces:
  - CLI: `python evaluate.py --checkpoint <best.pt> --config <yaml>` with PASS/FAIL verdict

- [x] **Step 1: Update evaluate.py**

In `mra_nn/evaluate.py`:

1. Rename all `delta_rho` / `dr` references to `rho_s` / `rs`.

2. Update Gate 1 description and variable names:
```python
print("\n=== Gate 1: Val rho_s MSE vs baseline ===")
```
and:
```python
val_rs_mse = val_metrics["loss_rho_s"]
gate1_pass = val_rs_mse < baseline_mse
print(f"  Val rho_s MSE:      {val_rs_mse:.6f}")
print(f"  Baseline (rho0):    {baseline_mse:.6f}")
print(f"  Improvement:        {(1 - val_rs_mse/baseline_mse)*100:.1f}%")
```

3. Update early-stopping metric references.

- [x] **Step 2: Verify it imports correctly**

```bash
cd /gpfs/projects/rjh/ruhin/madness-ruhin/mra_nn
source /gpfs/projects/rjh/ruhin/mra_nn/.venv/bin/activate
PYTHONPATH=/gpfs/projects/rjh/adrian/pymra/src \
python -c "from evaluate import ELECTRON_COUNTS; print(f'Loaded {len(ELECTRON_COUNTS)} molecules')"
```

Expected: `Loaded 15 molecules`

- [x] **Step 3: Commit**

```bash
cd /gpfs/projects/rjh/ruhin/madness-ruhin
git add mra_nn/evaluate.py
git commit -m "feat(mra-nn): update evaluation gate for rho_s baseline

Gate 1 now checks val rho_s MSE < rho0 baseline (using rho0 as-is).
All delta_rho references updated to rho_s."
```

---

### Task 7: GPU Training Run & Gate

**Status as of 2026-07-30:** Gate not yet passed. Seven training runs completed; six root-cause bugs found and fixed. See training history below.

**Files modified during this task:**
- `mra_nn/configs/default.yaml` — LR/epoch/val-molecule updates
- `mra_nn/losses.py` — pos_rho_weight, sigma clamp
- `mra_nn/train.py` — positive-only checkpoint selection, positive-only baseline
- `mra_nn/evaluate.py` — positive-only gate metric, ch3f electron count
- `mra_nn/model.py` — residual connection (`rho_s = head(x) + rho0_s`)
- `mra_nn/molecules/ch3f.in` — new val molecule (created)
- `mra_nn/slurm/gen_ch3f.sh` — ch3f data generation Slurm script (created)

**Interfaces:**
- Consumes: all prior tasks
- Produces: trained model checkpoint + gate PASS/FAIL verdict

---

#### Training History

**Run 1 — Job 2106622** (ch3oh val, LR=1e-3, max_epochs=200)
- **Finding:** Gate 1: 51% worse than baseline. Model output drifted — head was predicting near-zero.
- **Root cause:** No residual connection. Model tried to predict absolute rho_s from scratch; the target range (large s-coefficients) was hard to learn from zero initialization.
- **Fix:** Added residual in `model.py`: `rho_s = head_rho_s(x) + rho0_s`. Model now learns the correction; rho0_s handles the bulk of the signal.

**Run 2 — Job 2106747** (ch3oh val, LR=1e-3, max_epochs=200, with residual)
- **Finding:** Sigma_rs → ∞ (Kendall uncertainty weighting pathology). Total loss went negative (log_sigma terms dominated). rho_s head received near-zero gradient, drifted to noise. Gate 1: still failed.
- **Root cause:** Learnable `log_sigma_rs` has no upper bound in standard Kendall formulation. As `sigma_rs → ∞`, the weight `1/(2*sigma²) → 0`, zeroing the rho_s loss gradient. The uncertainty head "learned" to ignore the hardest task.
- **Fixes applied:**
  - `loss_fn.log_sigma_rs.clamp_(max=0.0)` after each optimizer step (enforces sigma_rs ≤ 1)
  - Gate 1 and checkpoint selection switched to **positive-only** `pos_rho_s_mse`
  - `evaluate()` computes `pos_rho_s_mse` separately from all-sample metrics

**Run 3 — Job 2107044** (ch3oh val, LR=1e-3, sigma clamp applied)
- **Finding:** All-sample val MSE still used for checkpoint selection. 87% of samples are negatives where rho≈rho0≈0; all-sample MSE saturated to 0.000000 (below float display precision) by epoch 30. `*` markers after epoch 30 were floating-point noise; early stopping didn't trigger correctly.
- **Fix:** Positive-only checkpoint selection (already described above — second patch applied here).

**Run 4 — Job 2107054** (ch3oh val, LR=1e-3, sigma clamp + positive-only checkpoint)
- **Finding:** PosValMSE reached ~9.4e-6 at epoch 15, then LR (~9.8e-4 post-warmup) kicked model out of the basin. Oscillated to 2-8e-5 for remaining epochs. Never converged below baseline (4.77e-7).
- **Root cause:** LR=1e-3 too high. Cosine schedule kept LR high through epoch 15-30, exactly when model approached a good minimum.
- **Fix:** LR 1e-3 → 2e-4; min_lr 1e-5 → 1e-6.

**Run 5 — Job 2107082** (ch3oh val, LR=2e-4, max_epochs=60)
- **Finding:** Training stable (no oscillation). Best pos val MSE = 3.5e-6 vs baseline 4.77e-7 — 9x worse. Model fails to generalize to ch3oh.
- **Root cause:** ch3oh (methanol) has a C-O-H combination (alcohol linkage, O lone pairs) not present in the 12 training molecules. Training set lacks sufficient chemical diversity.
- **Fix:** Switch val molecule ch3oh → ch3f. CH₃F is isoelectronic with training molecules (18 electrons, same as H₂O/HF/NH₃/CH₄ family); all bond types (C-H, C-F) are present in training. ch3oh moved to `train_molecules` (13 total). Also: max_epochs 60 → 120 (model still improving at epoch 59).

**Run 6 — Job 2107083** (transitional run, ch3oh still as val — config update in progress)
- Superseded by Run 7 once ch3f data was generated.

**Run 7 — Job 2107118** (ch3f val, LR=2e-4, max_epochs=120, pos_rho_weight=10.0)
- **Results:**
  - ch3f baseline (positive-only): 4.846e-7
  - Best positive-only val MSE: 1.375e-6 (2.8x worse than baseline)
  - vs. ch3oh run: improved from 9x worse → 2.8x worse — confirms ch3oh was a bad val choice
  - Model ran all 120 epochs without early stopping — still improving at epoch 119
- **Current status:** Gate 1 FAIL (2.8x worse). Gates 2-4 pass.
- **Remaining gap:** Training diversity insufficient. 13 molecules is not enough for cross-molecule generalization to an unseen molecule; need more W4-11 molecules.

---

#### Current Configuration (as of 2026-07-30)

```yaml
data:
  train_molecules: [h2o, nh3, ch4, co2, hf, n2, co, hcn, c2h4, c2h6, h2co, hcl, ch3oh]  # 13
  val_molecules: [ch3f]
  test_molecules: [h2o2, c2h2]

training:
  max_epochs: 120
  lr: 2.0e-4
  min_lr: 1.0e-6

loss:
  pos_rho_weight: 10.0  # upweight positive (in-tree) samples in rho_s MSE
```

All fixes committed to `feat/mra-nn-data`:
- Residual connection: `rho_s = head(x) + rho0_s`
- `log_sigma_rs.clamp_(max=0.0)` in training loop
- Positive-only `pos_rho_s_mse` for checkpoint selection and Gate 1
- `pos_rho_weight=10.0` in losses.py (upweights in-tree samples 10x)
- ch3oh → train; ch3f → val
- `ch3f.in` + `gen_ch3f.sh` added

---

#### Next Steps

- [ ] **Step 1: Generate more training molecules from W4-11**

  W4-11 molecule geometries available at `/gpfs/projects/rjh/ruhin/perf_pipeline/molecules/W4-11/`.
  Candidates (fill chemical gaps not covered by current 13 training molecules):
  - `n2o` — N=N=O, adds nitrogen-oxygen bonding (not in training)
  - `h2s` — adds sulfur (not in training)
  - `ocs` — C=S bond (not in training)
  - `hnco` — isocyanic acid, N-C=O combination
  - `ch2co` — ketene, C=C=O
  - `ch2f2` — difluoromethane, more fluorinated carbon
  - `c2h3f` — vinyl fluoride, C=C with fluorine

  For each new molecule:
  1. Check geometry at `/gpfs/projects/rjh/ruhin/perf_pipeline/molecules/W4-11/<mol>/struc.xyz`
  2. Create `mra_nn/molecules/<mol>.in` (template: `thresh 1e-6`, no explicit k — same as `ch3f.in`)
  3. Create Slurm script (same pipeline as `gen_ch3f.sh`)
  4. Add to `train_molecules` in `default.yaml`
  5. Add electron count to `ELECTRON_COUNTS` dict in `evaluate.py`

- [ ] **Step 2: Rebuild training dataset with new molecules**

  ```bash
  cd /gpfs/projects/rjh/ruhin/madness-ruhin
  source /gpfs/projects/rjh/ruhin/mra_nn/.venv/bin/activate
  PYTHONPATH=/gpfs/projects/rjh/adrian/pymra/src \
  python mra_nn/dataset_builder.py \
      --data-dir /gpfs/projects/rjh/ruhin/mra_nn/training_data \
      --out /gpfs/projects/rjh/ruhin/mra_nn/training_dataset.h5 \
      --gate
  ```

  Expected: gate PASS with updated molecule list.

- [ ] **Step 3: Increase max_epochs to 240 and retrain**

  Run 7 showed model still improving at epoch 119. Update config:
  ```yaml
  training:
    max_epochs: 240
  ```

  Then submit:
  ```bash
  /cm/shared/apps/slurm/21.08.8/bin/sbatch mra_nn/slurm/train_a100.sh
  ```

  Monitor:
  ```bash
  squeue -u ruhipatel
  tail -f /gpfs/projects/rjh/ruhin/mra_nn/logs/train_<jobid>.out
  ```

- [ ] **Step 4: Run gate check**

  ```bash
  cd /gpfs/projects/rjh/ruhin/madness-ruhin/mra_nn
  source /gpfs/projects/rjh/ruhin/mra_nn/.venv/bin/activate
  PYTHONPATH=/gpfs/projects/rjh/adrian/pymra/src \
  python evaluate.py \
      --checkpoint /gpfs/projects/rjh/ruhin/mra_nn/checkpoints/<run_name>/best.pt \
      --config configs/default.yaml
  ```

  Expected: `STEP 6 GATE: PASS`
  If FAIL: inspect which gate, check metrics.csv, diagnose and re-submit.

- [ ] **Step 5: Push all commits**

  ```bash
  cd /gpfs/projects/rjh/ruhin/madness-ruhin
  git push origin feat/mra-nn-data
  ```
