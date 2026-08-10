# MRA-NN Option B: Parent Node Features

**Date:** 2026-08-10
**Context:** Option A complete. Level-clamped model (10-14) gives 0.971x on val but same SCF iteration count. Coarse levels 1-9 are at parity because the model has no cross-level context. Parent features address this.

---

## Problem

The model sees only same-level information: center rho0/vnuc s-coefficients and 6 face-adjacent halo neighbors. At coarse levels (1-9), the density correction is inherently multi-scale — charge redistribution between atoms requires knowing the coarse structure of the region. Same-level halos cannot provide this.

SCF convergence is determined by coarse levels. The model adds nothing at levels 1-9, so even a perfect fine-level prediction (10-14) doesn't reduce iterations.

## Design

### Approach: Concatenate Parent S-Coefficients to Trunk Input

For each node at Key(n, l), extract the parent's rho0 and vnuc s-coefficients at Key(n-1, l//2) using `key.parent()` from pymra. Concatenate these 1024 floats (512 + 512) directly to the trunk input alongside the existing center + halo features.

**Why concatenation over a separate encoder or FiLM conditioning:**
- Simplest change — one new field in HDF5, one wider linear layer
- The trunk's FiLM layers already adapt processing by level
- No new modules to design, test, or tune
- If raw concatenation is too wide, a compressor can be added later without changing the data pipeline

### HDF5 Schema Changes

Two new fields per sample in `dataset_builder.py`:

| Field | Shape | dtype | Description |
|-------|-------|-------|-------------|
| `parent_rho0_s` | [N, k^3] | float32 | Parent node rho0 s-coefficients |
| `parent_vnuc_s` | [N, k^3] | float32 | Parent node vnuc s-coefficients |

**Level 0 handling:** No parent exists (n-1 = -1). Zero-pad both fields. The model already has level embedding via FiLM — level 0 gets its own learned embedding, so it can learn to ignore the zero parent features.

**Parent key computation:** `key.parent()` returns `Key(n-1, tuple(li >> 1 for li in l))`. The parent node always exists in the rho0/vnuc trees because MADNESS trees are complete from root to leaves — `node_s(tree, parent_key)` reconstructs coefficients at any ancestor via the two-scale relation.

### Model Changes

**Trunk input dimension:** 1792 -> 2816

Current: `halo_emb(768) + center(1024) = 1792`
New: `halo_emb(768) + center(1024) + parent(1024) = 2816`

Only the first FiLM layer's `nn.Linear(in_dim, 1024)` changes shape: `nn.Linear(2816, 1024)` instead of `nn.Linear(1792, 1024)`. All subsequent layers unchanged.

**Parameter increase:** ~1M new params in the first trunk layer (2816*1024 - 1792*1024 = 1,048,576 weights). Total model: ~3.04M -> ~4.09M.

**Forward pass change:**
```python
# New: parent features concatenated with center
center = torch.cat([rho0_s, vnuc_s], dim=-1)        # [B, 1024]
parent = torch.cat([parent_rho0_s, parent_vnuc_s], dim=-1)  # [B, 1024]
x = torch.cat([halo_emb, center, parent], dim=-1)   # [B, 2816]
```

### Dataset Loading Changes

`MRADataset.FIELD_NAMES` gains two entries: `parent_rho0_s`, `parent_vnuc_s`. Loaded as float32 tensors [N, 512]. Passed through to model forward.

### Inference Changes

`predict.py` `predict_density_simple()` must extract parent features at inference time, same as dataset_builder does at build time. For each leaf node, compute `key.parent()` and call `node_s(rho0_tree, parent_key)` / `node_s(vnuc_tree, parent_key)`.

### Config Changes

`single_task.yaml` model section:
- Add `use_parent_features: true`
- No other config changes needed — trunk_dims stay the same, the input widening is computed from the flag

### Training

- Same loss, same hyperparameters, same molecule split
- Train from scratch (checkpoint incompatible due to first layer shape)
- Same A100 job, same 120 epochs + patience 20
- Level clamping stays in inference (--use-model-levels)

---

## Success Criteria

1. Per-level ratio < 1.0 at levels 8-14 (currently only 10-14 beat baseline)
2. Per-level ratio < 1.0 at some levels in 1-7 range (currently all at parity)
3. Overall level-clamped ratio < 0.95x on val (currently 0.971x)
4. SCF iterations reduced vs. baseline on ch3oh

## Decision Gate

If levels 1-7 remain at parity after parent features: the correction requires context beyond one parent. Options:
- Message-passing on the tree (GNN-style, expensive to implement)
- Tree structure prediction pivot (Approach C from the original spec)

---

## Files Modified

| File | Change |
|------|--------|
| `mra_nn/dataset_builder.py` | Extract parent rho0/vnuc s-coefficients, add to HDF5 |
| `mra_nn/dataset.py` | Load `parent_rho0_s`, `parent_vnuc_s` fields |
| `mra_nn/model.py` | Accept parent features, widen trunk input |
| `mra_nn/predict.py` | Extract parent features at inference time |
| `mra_nn/configs/single_task.yaml` | Add `use_parent_features: true` |

No C++ changes. No pymra changes. No new dependencies.
