# MRA-NN Single-Task Ablation — Design Spec

**Date:** 2026-07-31
**Author:** Ruhi Patel + Claude (Opus 4.6)
**Status:** Draft
**Branch:** `feat/mra-nn-data`
**Parent spec:** `2026-07-27-mra-nn-step6-model-training-design.md`

---

## 1. Motivation

### What the diagnostic showed

On 2026-07-31 we ran `diagnose.py` against the best checkpoint (epoch 119, job 2107449) to measure whether the model beats the rho0 baseline on its own training data:

```
Split    Baseline (rho0)       Model MSE    Ratio   Beats?
-----------------------------------------------------------------
Train          3.811e-07       1.141e-06    2.99x       NO
Val            4.846e-07       1.375e-06    2.84x       NO
```

The model is 3x worse than rho0 on the molecules it trained on. Train and val ratios are nearly identical (2.99x vs 2.84x), ruling out overfitting. The head is producing corrections that are noise — further from the true correction Δ than zero is — even on seen data.

### Why "add more molecules" is the wrong next step

The Step 6 spec (section 10) proposed adding W4-11 molecules. More data helps generalization (closing a train/val gap). But there is no train/val gap — the model fails uniformly. Adding molecules to a model that can't fit its existing data is wasted compute.

### The hypothesis being tested

The current model shares a 3-layer trunk across three output heads:

| Head | Task | Performance |
|------|------|-------------|
| rho_s | Density coefficient regression (512-dim) | 3x worse than baseline |
| log_dnorm | Wavelet norm regression (scalar) | Converged (train loss ~0.48) |
| refine | Refinement classification (binary) | F1 = 0.87 |

The refine and log_dnorm heads work well. The rho_s head does not. One explanation: the shared trunk learns representations optimized for the easier tasks (classification + scalar regression) at the expense of the harder task (512-dim coefficient regression). The uncertainty weighting (Kendall) should balance this, but sigma_rs had to be clamped to prevent it going to infinity — a sign the optimization was actively trying to suppress the rho_s gradient.

**This ablation isolates the variable:** if a single-task model (rho_s only) beats the baseline on train data, multi-task interference is confirmed. If it still can't, the problem is elsewhere (architecture, input features, or the correction itself).

---

## 2. What Changes

This is a minimal ablation — change as few variables as possible to isolate multi-task interference.

### Removed
- log_dnorm output head
- refine output head
- `UncertaintyWeightedLoss` (no task weighting needed with one task)
- Focal loss
- Learnable sigma parameters
- Refine F1 metric tracking
- sigma clamping hack (`log_sigma_rs.clamp_(max=0.0)`)

### Kept identical
- Halo encoder architecture (shared, factored, 6 faces)
- FiLM conditioning (level embedding → per-layer gamma/beta)
- Trunk MLP (1792 → 1024 → 512 → 256, 3 layers, ReLU, dropout=0.1)
- rho_s output head (Linear(256, 512) + rho0_s residual)
- All hyperparameters: LR=2e-4, weight_decay=1e-4, cosine schedule, warmup=5, batch_size=4096
- Dataset: same 13 train molecules, ch3f val, same HDF5 file
- Data loading: same WeightedRandomSampler (10x on refine=1 positives)
- Positive-only MSE as checkpoint selection metric
- 120 epochs, patience=20

### Changed
- Loss: plain MSE on rho_s with 10x positive sample weighting (replaces uncertainty-weighted multi-task loss)
- Model forward returns only rho_s (not a 3-tuple)
- Training loop simplified — no refine metrics, no sigma logging

---

## 3. Architecture (ablated)

```
Input:  rho0_s(512) + vnuc_s(512) + 6x[halo_rho0(512) + halo_vnuc(512)] + level
                                    |
                    ┌───────────────┼───────────────┐
                    v               v               v
            Halo Encoder      Center Block     Level Embedding
           (shared, x6)       (1024-dim)        (32-dim)
            -> 6x128             |                 |
            concat=768           |          FiLM γ,β at each layer
                    |            |               |
                    └──────> Trunk MLP <─────────┘
                         1792 -> 1024 -> 512 -> 256
                              (3 FiLM layers)
                                    |
                                    v
                                 rho_s
                               (256->512)
                             linear + rho0_s
```

Parameter count drops from ~3.04M to ~2.91M (removing two small linear heads + 3 sigma scalars). Negligible difference — the trunk dominates.

---

## 4. Loss Function

```python
def single_task_loss(batch, pred_rho_s, pos_rho_weight=10.0):
    """MSE on rho_s with 10x weight on positive (in-tree) samples."""
    target = batch["rho_s"]
    neg_mask = batch["negative"]  # 1.0 for negative, 0.0 for positive

    # Per-sample weight: 10.0 for positive, 1.0 for negative
    weight = torch.where(neg_mask == 0, pos_rho_weight, 1.0)

    # Weighted MSE averaged over samples and 512 coefficients
    per_sample_mse = (pred_rho_s - target).pow(2).mean(dim=1)  # [B]
    loss = (weight * per_sample_mse).sum() / weight.sum()

    return loss
```

No learnable parameters in the loss. No sigma clamping needed.

---

## 5. Success Criteria

| Outcome | Train pos MSE vs baseline | Meaning | Next step |
|---------|--------------------------|---------|-----------|
| **A: Single-task beats baseline on train** | < 3.811e-7 | Multi-task interference confirmed | Re-introduce heads with split architecture (separate trunks or late branching) |
| **B: Single-task beats baseline on train AND val** | Train < 3.811e-7, Val < 4.846e-7 | Multi-task was the sole bottleneck | Re-introduce heads carefully; possibly proceed to Step 7 gate |
| **C: Single-task still fails on train** | > 3.811e-7 | Problem is not multi-task — architecture or input features insufficient | Investigate: (1) is the correction learnable at all from these features? (2) gradient analysis on head output |

The diagnostic script (`diagnose.py`) already exists and will be reused to evaluate the ablation checkpoint.

---

## 6. Implementation Approach

### Option 1: Config flag in existing files (Recommended)

Add `single_task: true` to config YAML. Model, loss, and training loop check this flag.

**Pros:** Same codebase, easy A/B comparison, git diff shows exactly what changed.
**Cons:** Adds conditional branches to existing code.

### Option 2: Separate model/loss/train files

Create `model_st.py`, `losses_st.py`, `train_st.py` for the ablation.

**Pros:** Clean separation; original code untouched.
**Cons:** Code duplication; divergence risk if shared logic changes.

**Selected: Option 1.** The conditionals are minimal (3-4 `if` checks), and keeping one codebase makes it easy to compare runs.

---

## 7. File Changes

| File | Change | Scope |
|------|--------|-------|
| `model.py` | Add `single_task` param to `MRANet.__init__`. When true, skip log_dnorm and refine heads. `forward()` returns `(rho_s, None, None)` to keep the 3-tuple interface — callers check for None. | ~15 lines |
| `losses.py` | Add `SingleTaskLoss` class (weighted MSE, no learnable params). | ~20 lines |
| `train.py` | Check `cfg["model"]["single_task"]`. Use `SingleTaskLoss` instead of `UncertaintyWeightedLoss`. Skip sigma clamping, refine F1, sigma logging. Adjust CSV columns. | ~30 lines |
| `configs/single_task.yaml` | Copy of `default.yaml` with `single_task: true` under `model:` and loss section trimmed to just `pos_rho_weight: 10.0`. | New file |
| `diagnose.py` | Already works — loads any checkpoint and evaluates rho_s. No changes needed. | None |

No changes to `dataset.py`, `predict.py`, or `evaluate.py`. The dataset still loads all fields (the unused ones are just ignored by the model).

---

## 8. Execution

1. Implement changes (Option 1)
2. Submit training job on A100 with `configs/single_task.yaml` (~2 hours for 120 epochs)
3. Run `diagnose.py` on the resulting checkpoint
4. Compare train/val pos MSE against the multi-task baseline:
   - Multi-task: Train 2.99x, Val 2.84x (from diagnostic run 2107449)
   - Single-task: target < 1.0x on train

---

## 9. What This Does NOT Test

To keep the ablation clean, the following are explicitly out of scope:

- **Removing the residual connection** — separate variable; test independently if needed
- **Changing the architecture** (deeper trunk, attention, etc.) — only relevant if single-task also fails (Outcome C)
- **Adding molecules** — only relevant after the model can fit existing data
- **Hyperparameter sweeps** — same LR/schedule/etc. for fair comparison
- **Head initialization changes** — the single-task model uses the same default PyTorch init
