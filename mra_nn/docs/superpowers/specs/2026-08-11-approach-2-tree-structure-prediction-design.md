# Approach 2: Tree Structure Prediction (Refinement Classification)

**Date:** 2026-08-11
**Context:** Density coefficient prediction is exhausted for SCF acceleration (Decisions 22-24). Fine-level density improvements have zero SCF iteration impact (14/14 multi-molecule tests). This spec pivots the ML role from density regression to tree refinement classification.

## Problem Statement

During each MADNESS SCF iteration, the solver walks every leaf node and computes wavelet coefficients to decide whether to refine (add children) or keep the node as a leaf. This adaptive refinement is a significant per-iteration cost. A model that accurately predicts refinement decisions from cheap features (rho0, vnuc, halos) could:

1. Skip expensive wavelet-norm computations for nodes confidently predicted as leaves
2. Pre-refine nodes predicted as needing children, avoiding iterative tree discovery
3. Build the correct tree topology upfront, reducing the number of refinement sweeps

## What Already Exists

All infrastructure for tree structure prediction is built and tested:

| Component | File | Status |
|-----------|------|--------|
| `refine` label (int8, 1=internal, 0=leaf/negative) | dataset_builder.py | Done |
| `head_refine` (Linear(256,1) logit) | model.py | Done |
| `FocalLoss` (gamma=2, alpha=0.75) | losses.py | Done |
| `UncertaintyWeightedLoss` (3-head balancing) | losses.py | Done |
| Multi-task training loop with refine_f1 tracking | train.py | Done |
| Tree-walk inference using refine head | predict.py `predict_density()` | Done |
| `WeightedRandomSampler` with refine_pos_weight | dataset.py | Done |
| 51-molecule HDF5 dataset with refine labels | training_dataset.h5 | Done |
| `default.yaml` multi-task config | configs/ | Needs molecule list update |

## Changes Required

### 1. New Config: `refine_task.yaml`

Based on `single_task.yaml`'s 51-molecule train/val/test split, but with multi-task training enabled:

- `single_task: false` — enables all three heads (rho_s, log_dnorm, refine)
- `use_parent_features: false` — parent features had no effect (Decision 22), remove them
- `refine_focused: true` — new flag to select best checkpoint by refine_f1 instead of pos_rho_s_mse

Same training hyperparameters: batch_size=4096, lr=2e-4, 120 epochs, patience=20.

### 2. Training Loop: Refine-Focused Checkpointing

In `train.py`, when `cfg.get("refine_focused", False)` is true:
- Best checkpoint selected by highest `refine_f1` (not lowest `pos_rho_s_mse`)
- Early stopping patience still based on the gate metric (refine_f1 improvement)
- All three losses still trained (rho_s and log_dnorm act as auxiliary regularizers)

### 3. Diagnostic Script: `diagnose_refine.py`

Per-level evaluation of the refine head:
- Precision, recall, F1 at each level (0-18)
- Confusion matrix at the decision boundary (levels 8-14)
- Overall accuracy on positive (in-tree) vs negative (below-leaf) samples
- Per-molecule refine F1 for val/test molecules
- Tree leaf count comparison: predicted vs true for each val molecule

### 4. SCF Integration Test

Update the SCF test to use tree-walk mode (`predict_density`) instead of copy mode (`predict_density_simple`). The model builds the complete tree from its refine predictions, fills leaves with predicted rho_s, then the resulting density is used as the SCF initial guess.

This tests whether the model can reconstruct the correct tree structure well enough for SCF to converge.

## Architecture

No changes to the model architecture. The existing MRANet with three heads:

```
Input: rho0_s[512] + vnuc_s[512] + halo_rho0[6×512] + halo_vnuc[6×512] + level[1]
  → HaloEncoder → [768]
  → concat(halo_emb[768], center[1024]) → [1792]
  → FiLM trunk (1792→1024→512→256) conditioned on level_embedding[32]
  → head_rho_s: Linear(256, 512) + residual from rho0_s
  → head_log_dnorm: Linear(256, 1)
  → head_refine: Linear(256, 1) → sigmoid → P(refine)
```

3.04M parameters (same as Option A). The `UncertaintyWeightedLoss` auto-balances the three tasks via learned log-sigma parameters.

## Training Data

Same 51-molecule HDF5 dataset already built (5.29 GB). Class balance for refine labels:

- `refine=1` (internal nodes, needs refinement): ~13% of all samples
- `refine=0` (leaf + negative/below-leaf): ~87% of all samples
- Among in-tree nodes only (`negative=0`): roughly 50/50 refine vs leaf
- `WeightedRandomSampler` with `refine_pos_weight=10.0` compensates for imbalance

## Success Criteria

1. **Refine F1 > 0.95** on val molecules (ethanol, so2, hnnn)
2. **Per-level accuracy > 90%** at levels 8-14 (the refinement decision boundary)
3. **Tree structure similarity** — predicted tree leaf count within 10% of true for val molecules
4. **SCF convergence** — moldft with ML-predicted tree (tree-walk mode) converges to correct ground state (energy within 1e-6 Ha of baseline)

## What This Does NOT Include

- C++ integration to accelerate MADNESS refinement at runtime (future work)
- Changes to the model architecture
- Changes to the dataset or dataset builder
- Parent features (proven ineffective)

## Risk Assessment

**Low risk:** The model already works well at fine levels for density prediction. Refinement classification is a strictly easier task (binary vs 512-dim regression). The infrastructure is fully built.

**Main uncertainty:** Whether the uncertainty-weighted multi-task loss properly balances the refine head against the density heads with 51 molecules. The original multi-task training used only 13 molecules. If the refine head underperforms, a refine-only single-task model (dropping rho_s and log_dnorm heads entirely) is a straightforward fallback.
