# Approach 2: Tree Structure Prediction — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Use **Sonnet** for all tasks — these are mechanical config/script changes on existing infrastructure.

**Goal:** Train the existing multi-task MRANet with refine-focused checkpointing on 51 molecules, evaluate refinement accuracy per-level, and run an SCF integration test using tree-walk inference.

**Architecture:** Reuse existing multi-task model (3 heads: rho_s, log_dnorm, refine) with UncertaintyWeightedLoss. Change the best-checkpoint metric from pos_rho_s_mse to refine_f1. Add a refine diagnostic script and a tree-walk SCF test.

**Tech Stack:** PyTorch, pymra, MADNESS (moldft), HDF5, Slurm

## Global Constraints

- All code lives in `/gpfs/projects/rjh/ruhin/madness-ruhin/mra_nn/` on branch `feat/mra-nn-data`
- Working data dir: `/gpfs/projects/rjh/ruhin/mra_nn/` (not in git)
- Python venv: `/gpfs/projects/rjh/ruhin/mra_nn/.venv/bin/activate`
- PYTHONPATH must include `/gpfs/projects/rjh/ruhin/madness-ruhin:/gpfs/projects/rjh/adrian/pymra/src`
- Dataset: `/gpfs/projects/rjh/ruhin/mra_nn/training_dataset.h5` (51 molecules, 5.29 GB, already has refine labels)
- Model architecture: do NOT change model.py — use existing MRANet with `single_task: false`
- No Co-Authored-By lines in commits
- MAD_NUM_THREADS must be set to ntasks - 1
- Slurm binary: `/cm/shared/apps/slurm/21.08.8/bin/sbatch`

---

### Task 1: Config + Training Loop Changes

**Files:**
- Create: `mra_nn/configs/refine_task.yaml`
- Modify: `mra_nn/train.py:278-320` (checkpointing logic)

**Interfaces:**
- Consumes: existing `single_task.yaml` molecule lists, `UncertaintyWeightedLoss`, `build_model(cfg)`
- Produces: `refine_task.yaml` config consumed by train.py; `refine_focused` flag in train.py checkpointing

- [ ] **Step 1: Create refine_task.yaml**

```yaml
# MRA-NN Approach 2 — refine-focused multi-task config
# Uses 51-molecule split from single_task.yaml with multi-task training enabled.
# Gate metric: refine_f1 (not pos_rho_s_mse).

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
    - ch3f
    - f2
    - cl2
    - clf
    - h2s
    - cs2
    - hnc
    - hof
    - hocl
    - n2o
    - ocs
    - so3
    - f2o
    - cl2o
    - clcn
    - hno
    - hcno
    - hnco
    - hocn
    - hcof
    - c-n2h2
    - t-n2h2
    - ch2f2
    - cf4
    - nh2cl
    - n2h4
    - ch3nh2
    - ketene
    - formic
    - acetaldehyde
    - allene
    - oxirane
  val_molecules:
    - ethanol
    - so2
    - hnnn
  test_molecules:
    - h2o2
    - c2h2
    - glyoxal

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
  single_task: false
  use_parent_features: false

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
  focal_gamma: 2.0
  focal_alpha: 0.75
  refine_pos_weight: 10.0
  pos_rho_weight: 10.0

refine_focused: true

checkpoint:
  dir: /gpfs/projects/rjh/ruhin/mra_nn/checkpoints
```

- [ ] **Step 2: Modify train.py checkpointing for refine_focused mode**

In `main()`, after `single_task = cfg["model"].get("single_task", False)` (around line 197), add:

```python
refine_focused = cfg.get("refine_focused", False)
```

Replace the checkpointing block (lines ~313-320) — currently:

```python
is_best = val_metrics["pos_rho_s_mse"] < best_val_rs_mse
if is_best:
    best_val_rs_mse = val_metrics["pos_rho_s_mse"]
    patience_counter = 0
else:
    patience_counter += 1
```

With:

```python
if refine_focused and not single_task:
    # Gate on refine F1 (higher is better)
    current_gate = val_metrics.get("refine_f1", 0.0)
    is_best = current_gate > best_gate_value
else:
    # Gate on positive rho_s MSE (lower is better)
    current_gate = val_metrics["pos_rho_s_mse"]
    is_best = current_gate < best_gate_value

if is_best:
    best_gate_value = current_gate
    patience_counter = 0
else:
    patience_counter += 1
```

Also rename `best_val_rs_mse` to `best_gate_value` at initialization (line ~280):

```python
best_gate_value = 0.0 if (refine_focused and not single_task) else float("inf")
```

Update the final summary (lines ~362-368):

```python
if refine_focused and not single_task:
    print(f"  Best val refine F1: {best_gate_value:.4f}")
    print(f"  Gate: {'PASS (F1 > 0.95)' if best_gate_value > 0.95 else 'FAIL'}")
else:
    print(f"  Best pos val rho_s MSE: {best_gate_value:.3e}")
    print(f"  Baseline (pos, rho0):   {baseline_mse:.3e}")
    if best_gate_value < baseline_mse:
        print(f"  Model BEATS baseline by {(1 - best_gate_value/baseline_mse)*100:.1f}%")
    else:
        print(f"  Model DOES NOT beat baseline")
```

- [ ] **Step 3: Verify train.py loads correctly**

```bash
cd /gpfs/projects/rjh/ruhin/madness-ruhin
source /gpfs/projects/rjh/ruhin/mra_nn/.venv/bin/activate
export PYTHONPATH=/gpfs/projects/rjh/ruhin/madness-ruhin:/gpfs/projects/rjh/adrian/pymra/src
python -c "from mra_nn.train import main; print('train.py imports OK')"
```

Expected: `train.py imports OK`

- [ ] **Step 4: Commit**

```bash
git add mra_nn/configs/refine_task.yaml mra_nn/train.py
git commit -m "Add refine-focused config and checkpointing for Approach 2"
```

---

### Task 2: Refine Diagnostic Script

**Files:**
- Create: `mra_nn/diagnose_refine.py`

**Interfaces:**
- Consumes: trained checkpoint (best.pt), `refine_task.yaml` config, `MRADataset`, `build_model`
- Produces: per-level precision/recall/F1 table, per-molecule refine F1, confusion matrix

- [ ] **Step 1: Write diagnose_refine.py**

```python
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
    """Evaluate refine head on a dataset. Returns per-level metrics."""
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
```

- [ ] **Step 2: Verify it imports**

```bash
python -c "import mra_nn.diagnose_refine; print('OK')" 2>&1 || \
python diagnose_refine.py --help
```

- [ ] **Step 3: Commit**

```bash
git add mra_nn/diagnose_refine.py
git commit -m "Add per-level refine diagnostic script for Approach 2"
```

---

### Task 3: Training Job Submission

**Files:**
- Create: `mra_nn/slurm/train_refine.sh`

**Interfaces:**
- Consumes: `refine_task.yaml`, `train.py` with refine_focused support
- Produces: trained checkpoint in `/gpfs/projects/rjh/ruhin/mra_nn/checkpoints/<timestamp>/best.pt`

- [ ] **Step 1: Create train_refine.sh**

```bash
#!/bin/bash
#SBATCH --partition=a100-long
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --job-name=mra-ref
#SBATCH --output=/gpfs/projects/rjh/ruhin/mra_nn/logs/train_refine_%j.out
#SBATCH --error=/gpfs/projects/rjh/ruhin/mra_nn/logs/train_refine_%j.err

set -euo pipefail

source /gpfs/projects/rjh/ruhin/mra_nn/.venv/bin/activate
export PYTHONPATH=/gpfs/projects/rjh/ruhin/madness-ruhin:/gpfs/projects/rjh/adrian/pymra/src:${PYTHONPATH:-}

cd /gpfs/projects/rjh/ruhin/madness-ruhin

python mra_nn/train.py --config mra_nn/configs/refine_task.yaml
```

- [ ] **Step 2: Validate and submit**

```bash
/cm/shared/apps/slurm/21.08.8/bin/sbatch --test-only mra_nn/slurm/train_refine.sh
/cm/shared/apps/slurm/21.08.8/bin/sbatch mra_nn/slurm/train_refine.sh
```

- [ ] **Step 3: Commit**

```bash
git add mra_nn/slurm/train_refine.sh
git commit -m "Add Slurm script for refine-focused multi-task training"
```

---

### Task 4: Evaluate and SCF Test (after training completes)

**Files:**
- Create: `mra_nn/slurm/scf_test_treewalk.sh`

**Interfaces:**
- Consumes: trained checkpoint from Task 3, `predict.py` `predict_density()` (tree-walk mode), `diagnose_refine.py` from Task 2
- Produces: per-level refine accuracy, SCF convergence comparison

- [ ] **Step 1: Run diagnose_refine.py on the trained checkpoint**

```bash
cd /gpfs/projects/rjh/ruhin/madness-ruhin
source /gpfs/projects/rjh/ruhin/mra_nn/.venv/bin/activate
export PYTHONPATH=/gpfs/projects/rjh/ruhin/madness-ruhin:/gpfs/projects/rjh/adrian/pymra/src

python mra_nn/diagnose_refine.py \
    --checkpoint /gpfs/projects/rjh/ruhin/mra_nn/checkpoints/<TIMESTAMP>/best.pt \
    --config mra_nn/configs/refine_task.yaml
```

Check output for:
- Overall val refine F1 > 0.95
- Per-level accuracy > 90% at levels 8-14

- [ ] **Step 2: Create scf_test_treewalk.sh**

This script tests SCF convergence using tree-walk inference (model builds tree from scratch using refine head predictions) on ch3oh as a reference molecule.

```bash
#!/bin/bash
# SCF test using tree-walk inference (Approach 2)
# The model predicts BOTH tree structure (refine) AND density coefficients.
#
#SBATCH --partition=long-40core
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH --job-name=scf-tree
#SBATCH --output=/gpfs/projects/rjh/ruhin/mra_nn/logs/scf_treewalk_%j.out
#SBATCH --error=/gpfs/projects/rjh/ruhin/mra_nn/logs/scf_treewalk_%j.err

source /etc/profile.d/modules.sh
module load shared gcc/13.2.0

INTEL_PATH="/gpfs/software/intel/oneAPI/2024_2"
source "${INTEL_PATH}/setvars.sh" --force
export LD_LIBRARY_PATH=${INTEL_PATH}/tbb/2021.13/lib/intel64/gcc4.8:${LD_LIBRARY_PATH:-}
export PATH=${INTEL_PATH}/mpi/2021.13/bin:/cm/shared/apps/slurm/21.08.8/bin:$PATH

set -euo pipefail

BUILD=/gpfs/projects/rjh/ruhin/madness-build-hdf5
SRC=/gpfs/projects/rjh/ruhin/madness-ruhin
DATA=/gpfs/projects/rjh/ruhin/mra_nn
CXX=/gpfs/software/gcc/13.2.0/bin/g++
HDF5_ROOT=/cm/shared/apps/hdf5/1.12.1
PYTHON=$DATA/.venv/bin/python

export PYTHONPATH=/gpfs/projects/rjh/ruhin/madness-ruhin:/gpfs/projects/rjh/adrian/pymra/src:${PYTHONPATH:-}
export MAD_NUM_THREADS=0

# UPDATE THIS with the actual checkpoint path after training
CHECKPOINT=$DATA/checkpoints/REPLACE_WITH_TIMESTAMP/best.pt
MOL=ch3oh
NELEC=18

TESTDIR=$DATA/scf_test_treewalk/$MOL
mkdir -p "$TESTDIR"

# ============================================================================
# Step 0: Build binaries (same as scf_test.sh)
# ============================================================================
echo "========================================="
echo "Step 0: Building binaries on this node"
echo "========================================="
cd "$BUILD"

CXXFLAGS="-march=native -std=c++17 -O3 -DNDEBUG -fPIC"
DEFINES="-DMADNESS_LINALG_USE_LAPACKE \
  -DMADNESS_MPI_HEADER=\"${INTEL_PATH}/mpi/2021.13/include/mpi.h\" \
  -DMAD_ROOT_DIR=\"$SRC\" \
  -DMRA_CHEMDATA_DIR=\"$SRC/src/madness/chem\""
INCLUDES="-I$BUILD/src/madness/chem -I$SRC/src/madness/chem \
  -I$SRC/src -I$BUILD/src \
  -I$BUILD/src/madness/world -I$SRC/src/madness/world \
  -I${INTEL_PATH}/tbb/2021.13/include -I${INTEL_PATH}/mpi/2021.13/include \
  -I$BUILD/src/madness/misc -I$SRC/src/madness/misc \
  -I${INTEL_PATH}/mkl/2024.2/include \
  -I$BUILD/src/madness/tensor -I$SRC/src/madness/tensor \
  -I$BUILD/src/madness/external/tinyxml -I$SRC/src/madness/external/tinyxml \
  -I$BUILD/src/madness/external/muParser -I$SRC/src/madness/external/muParser \
  -I$BUILD/src/madness/mra -I$SRC/src/madness/mra \
  -I$SRC/src/apps -I$BUILD -I$SRC"
LINK_FLAGS="-Xlinker --enable-new-dtags \
  -Xlinker -rpath -Xlinker ${INTEL_PATH}/mpi/2021.13/lib \
  -Xlinker -rpath -Xlinker ${INTEL_PATH}/mpi/2021.13/lib \
  -Xlinker --enable-new-dtags"
LIBS="$BUILD/src/madness/chem/libMADchem.a \
  $BUILD/src/madness/mra/libMADmra.a \
  $BUILD/src/madness/tensor/libMADlinalg.a \
  $BUILD/src/madness/tensor/libMADtensor.a \
  $BUILD/src/madness/misc/libMADmisc.a \
  $BUILD/src/madness/world/libMADworld.a \
  ${INTEL_PATH}/tbb/2021.13/lib/intel64/gcc4.8/libtbb.so.12 \
  ${INTEL_PATH}/mpi/2021.13/lib/libmpicxx.so \
  ${INTEL_PATH}/mpi/2021.13/lib/libmpi.so"
LINK_TAIL="-lrt -lpthread -ldl -Wl,--start-group \
  ${INTEL_PATH}/mkl/2024.2/lib/libmkl_intel_lp64.a \
  ${INTEL_PATH}/mkl/2024.2/lib/libmkl_core.a \
  ${INTEL_PATH}/mkl/2024.2/lib/libmkl_sequential.a \
  -Wl,--end-group -lm -ldl \
  $BUILD/src/madness/external/tinyxml/libMADtinyxml.a \
  $BUILD/src/madness/external/muParser/libMADmuparser.a"

echo "  Compiling SCF.cc..."
$CXX $CXXFLAGS $DEFINES $INCLUDES \
  -c "$SRC/src/madness/chem/SCF.cc" \
  -o src/madness/chem/CMakeFiles/MADchem-obj.dir/SCF.cc.o

echo "  Rebuilding libMADchem.a..."
cd src/madness/chem
ar rcs libMADchem.a CMakeFiles/MADchem-obj.dir/*.o
cd "$BUILD"

echo "  Relinking moldft..."
$CXX $CXXFLAGS $LINK_FLAGS \
  src/apps/moldft/CMakeFiles/moldft.dir/moldft.cc.o \
  -o src/apps/moldft/moldft \
  $LIBS $LINK_TAIL

MOLDFT=$BUILD/src/apps/moldft/moldft
echo "  moldft OK"

echo "  Compiling h5_to_archive..."
H5_SRC="$SRC/src/apps/molresponse/tools/h5_to_archive.cpp"
H5_OUT="$BUILD/src/apps/molresponse/h5_to_archive"

$CXX $CXXFLAGS $DEFINES -DMADNESS_HAS_HDF5 \
  $INCLUDES \
  -I"$BUILD/src/apps/molresponse" -I"$SRC/src/apps/molresponse" \
  -I"$HDF5_ROOT/include" \
  -c "$H5_SRC" -o "${H5_OUT}.o"

$CXX $CXXFLAGS $LINK_FLAGS \
  "${H5_OUT}.o" -o "$H5_OUT" \
  -Wl,-rpath,"$HDF5_ROOT/lib" \
  $LIBS "$HDF5_ROOT/lib/libhdf5.so" -lz -ldl -lm $LINK_TAIL

H5_TO_ARCHIVE=$H5_OUT
echo "  h5_to_archive OK"
echo ""

# ============================================================================
# Step 1: Tree-walk prediction (model builds tree structure)
# ============================================================================
echo "========================================="
echo "Step 1: Tree-walk density prediction"
echo "========================================="
cd "$SRC"

# NOTE: No --use-model-levels flag — tree-walk mode uses refine head
$PYTHON mra_nn/predict.py \
    --checkpoint "$CHECKPOINT" \
    --rho0 "$DATA/training_data/${MOL}/rho0.mad.h5" \
    --vnuc "$DATA/training_data/${MOL}/vnuc.mad.h5" \
    --n-electrons $NELEC \
    --out "$TESTDIR/rhoML.mad.h5"

echo "  rhoML.mad.h5 written"

# ============================================================================
# Step 2: Convert HDF5 to MADNESS archive
# ============================================================================
echo ""
echo "========================================="
echo "Step 2: Converting HDF5 to MADNESS archive"
echo "========================================="
cd "$TESTDIR"
mpirun -np 1 "$H5_TO_ARCHIVE" rhoML.mad.h5 rhoML || true

if [ ! -f rhoML.00000 ]; then
    echo "ERROR: rhoML.00000 not written"
    exit 1
fi
echo "  rhoML.00000 written"

# ============================================================================
# Step 3: Baseline SCF
# ============================================================================
echo ""
echo "========================================="
echo "Step 3: Baseline SCF (rho0 initial guess)"
echo "========================================="
BASELINE_DIR="$TESTDIR/baseline"
mkdir -p "$BASELINE_DIR"
cp "$DATA/molecules/${MOL}.in" "$BASELINE_DIR/input"
cd "$BASELINE_DIR"
rm -f *.restartdata* rhoML.00000

mpirun -np 1 "$MOLDFT" 2>&1 | tee moldft_baseline.log

# ============================================================================
# Step 4: ML-guess SCF (tree-walk predicted density)
# ============================================================================
echo ""
echo "========================================="
echo "Step 4: ML-guess SCF (tree-walk prediction)"
echo "========================================="
ML_DIR="$TESTDIR/ml_guess"
mkdir -p "$ML_DIR"
cp "$DATA/molecules/${MOL}.in" "$ML_DIR/input"
cp "$TESTDIR/rhoML.00000" "$ML_DIR/rhoML.00000"
cd "$ML_DIR"
rm -f *.restartdata*

mpirun -np 1 "$MOLDFT" 2>&1 | tee moldft_ml.log

# ============================================================================
# Step 5: Compare
# ============================================================================
echo ""
echo "========================================="
echo "Step 5: Comparison"
echo "========================================="
echo ""
echo "=== BASELINE ==="
grep "final energy" "$BASELINE_DIR/moldft_baseline.log" || echo "(not found)"
BL_ITERS=$(grep -c "^Iteration" "$BASELINE_DIR/moldft_baseline.log" 2>/dev/null || echo "0")
echo "  Total iterations: $BL_ITERS"
echo ""
echo "=== ML GUESS (tree-walk) ==="
grep "final energy" "$ML_DIR/moldft_ml.log" || echo "(not found)"
ML_ITERS=$(grep -c "^Iteration" "$ML_DIR/moldft_ml.log" 2>/dev/null || echo "0")
echo "  Total iterations: $ML_ITERS"
echo ""
echo "========================================="
echo "SCF tree-walk test complete"
echo "========================================="
```

- [ ] **Step 3: Commit**

```bash
git add mra_nn/slurm/scf_test_treewalk.sh
git commit -m "Add tree-walk SCF test script for Approach 2"
```

- [ ] **Step 4: After training completes — run diagnostic and submit SCF test**

Update `CHECKPOINT` in `scf_test_treewalk.sh` with actual timestamp, then:

```bash
# Run diagnostic
python mra_nn/diagnose_refine.py \
    --checkpoint /gpfs/projects/rjh/ruhin/mra_nn/checkpoints/<TIMESTAMP>/best.pt \
    --config mra_nn/configs/refine_task.yaml

# Submit SCF test
/cm/shared/apps/slurm/21.08.8/bin/sbatch mra_nn/slurm/scf_test_treewalk.sh
```
