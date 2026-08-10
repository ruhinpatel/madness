# MRA-NN SCF Test Post-Mortem: Decision Audit and Path Forward

**Date:** 2026-08-10
**Context:** Option C SCF convergence test on ch3oh (methanol) — ML-predicted density vs promolecular density (rho0) as SCF initial guess in MADNESS moldft.

## Executive Summary

The ML-predicted density is **65% farther** from the converged SCF density than the simple promolecular guess (rho0). When used as an SCF initial guess, it leads to a **wrong electronic state** 0.587 Ha above the true ground state. The model makes predictions worse on **85% of tree leaves** (1555/1828).

Root causes are architectural (insufficient spatial context), training-related (multi-task interference, now resolved), and data-related (level sample imbalance). The path forward is Option A (more data + level masking) then Option B (parent node features).

---

## SCF Test Results (2026-08-09)

### What was built

1. **C++ injection point** in `SCF::initial_guess()` (SCF.cc, after line 1032): detects `rhoML.00000` archive, loads it via `ParallelInputArchive`, rescales to correct electron count. Falls back to normal rho0 if archive absent.

2. **h5_to_archive converter** (`src/apps/molresponse/tools/h5_to_archive.cpp`): reads `.mad.h5` structured HDF5, writes MADNESS binary archive via `ParallelOutputArchive`.

3. **End-to-end test script** (`mra_nn/slurm/scf_test.sh`): builds binaries on-node (avoids -march=native cross-architecture issues), runs ML prediction, converts HDF5 to archive, runs baseline moldft, runs ML-guess moldft, compares.

### Results

| Metric | Baseline (rho0) | ML guess (rhoML) |
|--------|-----------------|------------------|
| Final energy (Ha) | -114.850 | -114.263 |
| Protocol 1 iterations | 8 (converged) | 20 (maxiter, not converged) |
| Protocol 2 iterations | 2 (converged) | 10 (converged) |
| Total iterations | 10 | 30 |
| Dipole moment (a.u.) | 0.645 | 2.015 |

The ML-guess SCF converged to a **qualitatively different electronic state** — different orbital energies, 3x larger dipole moment pointing in the wrong direction.

### Density Comparison (`mra_nn/compare_densities.py`)

| Comparison | L2 norm | Relative error |
|------------|---------|----------------|
| rho0 vs rho_conv | 1.184 | 11.1% |
| **rhoML vs rho_conv** | **1.955** | **16.2%** |
| rho0 vs rhoML | 1.339 | 12.6% |

**Ratio (ML/rho0): 1.65** — ML is 65% farther from the converged density.

Per-leaf analysis on rho0's 1828 leaves:
- ML improves 273 leaves (15%)
- ML worsens 1555 leaves (85%)
- Per-leaf MSE: ML 4.05e-6 vs rho0 1.49e-6 (ML is 2.7x worse)

Dipole verification (validated against MADNESS SCF output):
- rho0 electronic dipole: ~0 (correct — spherical atomic superposition)
- rho_conv electronic dipole: (-0.309, -0.566, 0.000) — matches MADNESS baseline
- rhoML electronic dipole: (-4.826, -1.639, +4.038) — 10x too large, spurious z-component

---

## The 16 Decision Points

### Decision 1: Problem Formulation — MRA Coefficient Space
- **When:** Project inception (pre-July 2026)
- **What:** Predict density in MRA scaling coefficient space, not on a real-space grid
- **Why:** MRA tree encodes spatial structure; wavelet d-coefficients encode local error for free
- **Verdict:** Sound. Avoids grid-to-MRA projection loss.

### Decision 2: Training Molecules — 15 Small Molecules from W4-11
- **When:** 2026-07-16 (commit `d4aab4fe`)
- **What:** 15 molecules (H2O, NH3, CH4, CO2, HF, N2, CO, HCN, C2H2, C2H4, C2H6, H2CO, CH3OH, H2O2, HCl), 13 for training
- **Why:** W4-11 provides reliable geometries; small molecules are computationally manageable
- **Verdict:** Insufficient. Only ~207k samples total with extreme level imbalance. More molecules from W4-11 were planned but never added. This is a direct cause of data starvation at extreme tree levels.

### Decision 3: Data Pipeline — Three-Step MADNESS Dump
- **When:** 2026-07-16 (commit `d4aab4fe`)
- **What:** (1) `dump_training_functions` for rho0/vnuc, (2) `moldft` for SCF convergence, (3) `dump_training_functions --archive` for rho_conv
- **Why:** Reuses existing MADNESS tools
- **Verdict:** Sound. Pipeline works correctly.

### Decision 4: Dataset Construction — All Nodes + Below-Leaf Negatives
- **When:** 2026-07-27 (commit `8be3292`)
- **What:** Dataset has all rho_conv tree nodes (positive, refine=1 for internal, refine=0 for leaves) plus 8 children of each leaf (negative, refine=0)
- **Why:** The refine head needs both refinement and stop examples
- **Verdict:** Problematic. Created 87% negative class imbalance that dominated all metrics and gradients. Required multiple band-aids (pos_rho_weight, positive-only metrics, WeightedRandomSampler) that never fully resolved the issue.

### Decision 5: Architecture — FiLM MLP with 6 Same-Level Halo Neighbors
- **When:** 2026-07-28 (commit `4ae1c9c`)
- **What:** HaloEncoder processes 6 face-adjacent neighbors (shared MLP), concatenated with center features, fed through 3 FiLM-conditioned layers (1024->512->256). FiLM conditioning from level embedding.
- **Why:** Weight-efficient spatial context; FiLM adapts to box size
- **Verdict:** ROOT CAUSE #1. No parent features, no cross-level information. Model outputs corrections of right magnitude (0.66x of |delta|) but wrong direction — classic missing-information signature. Density redistribution is multi-scale; same-level neighbors alone are insufficient.

### Decision 6: Original Target — Delta-Rho (Correction)
- **When:** 2026-07-28 (commit `4ae1c9c`)
- **What:** Target was delta_rho = rho_conv_s - rho0_s
- **Verdict:** Superseded by Decision 9/10 (direct rho_s with residual connection). Not impactful.

### Decision 7: Loss — Kendall Uncertainty-Weighted Multi-Task
- **When:** 2026-07-28 (commit `4c93df3`)
- **What:** Three losses (rho_s MSE, log_dnorm MSE, refine focal loss) with learnable log-sigma auto-balancing per Kendall et al. 2018
- **Why:** Auto-balance heterogeneous multi-task losses
- **Verdict:** Failed. sigma_rs grew unbounded, zeroing the rho_s gradient. The optimizer learned to ignore the hardest task. Required clamping band-aid (Decision 12) that violated the auto-balancing premise.

### Decision 8: Hyperparameters — lr=1e-3, batch=4096
- **When:** 2026-07-28 (commit `1d63d3e`)
- **What:** AdamW, lr=1e-3, cosine schedule, warmup=5, patience=20
- **Verdict:** LR too high (caused oscillation). Fixed to 2e-4 on 2026-07-31.

### Decision 9: Switch to k=8 and Direct rho_s Target
- **When:** 2026-07-30 (commits `66d6d81`-`13b2778`)
- **What:** k=6/thresh=1e-4 to k=8/thresh=1e-6 (production precision). Target from delta_rho to direct rho_s.
- **Why:** Adrian directive — k=8 is MADNESS production standard
- **Verdict:** Necessary. k^3 went from 216 to 512 (~2.8M params).

### Decision 10: Residual Connection
- **When:** 2026-07-30 (commit `0bed888`)
- **What:** `rho_s = head(x) + rho0_s` — model learns the small correction
- **Why:** Without residual, model plateaued 1200x above baseline (couldn't learn near-identity mapping through FiLM layers)
- **Verdict:** Correct and essential.

### Decision 11: Positive Sample Weighting (10x)
- **When:** 2026-07-30 (commit `8b27798`)
- **What:** In rho_s MSE, positive samples (in-tree) weighted 10x vs negatives (below-leaf)
- **Why:** 87% negatives dominated gradient. 10x shifts gradient to ~75% positive.
- **Verdict:** Band-aid. Partially addressed Decision 4's imbalance. The 10x value was not tuned.

### Decision 12: Clamp log_sigma_rs <= 0
- **When:** 2026-07-30 (commit `9d4e7a1`)
- **What:** After each optimizer step, clamp sigma_rho_s <= 1
- **Why:** sigma_rs grew unbounded in Run 2, zeroing rho_s gradient
- **Verdict:** Band-aid on Decision 7. If you must clamp sigma to prevent task shutdown, the auto-balancing premise is violated.

### Decision 13: Positive-Only Checkpoint Selection
- **When:** 2026-07-31 (commit `f9910cf`)
- **What:** Best checkpoint selected by positive-only val MSE instead of all-sample MSE
- **Why:** All-sample MSE saturated to ~5e-8 (dominated by trivial negatives), making checkpoint selection random
- **Verdict:** Correct fix. But highlights how the 87% negative dominance warped all metrics.

### Decision 14: Validation Molecule Switch (CH3OH -> CH3F)
- **When:** 2026-07-31 (commit `11b2fd0`)
- **What:** CH3OH moved to training; CH3F became validation molecule
- **Why:** CH3OH had C-O-H bonds not in training. CH3F is isoelectronic (18e), bonds (C-H, C-F) appear in training (CH4, HF).
- **Verdict:** Masked the generalization problem. Choosing a validation molecule chemically close to training gave a false sense of progress (val 1.00x). A harder validation molecule would have surfaced issues earlier.

### Decision 15: Single-Task Ablation
- **When:** 2026-07-31 (commits `ed0c20e`-`b712554`)
- **What:** Removed log_dnorm and refine heads. Plain weighted MSE, no learnable sigmas.
- **Results:** Train ratio 2.99x -> 1.28x. Val ratio 2.84x -> 1.00x.
- **Verdict:** Most informative experiment. Proved multi-task interference was a major bottleneck but not the only one. Even single-task can't beat baseline due to insufficient context (root cause #1) and data starvation (root cause #3).

### Decision 16: Compress-Reconstruct in Inference
- **When:** 2026-08-05 (commit `5c0a055`)
- **What:** `predict_density_simple()` copies rho0 tree, replaces leaf coefficients, runs compress->reconstruct
- **Why:** Adrian directed: don't trust interior node predictions as coefficients; enforce two-scale consistency
- **Verdict:** Sound engineering choice.

---

## Three Root Causes (in order of severity)

### Root Cause #1: Insufficient Spatial Context (Architecture)

The model sees only 6 face-adjacent neighbors at the SAME tree level. No parent node features, no cross-level information. Electron density redistribution during SCF is inherently multi-scale — charge flows between atoms (coarse levels) and concentrates near nuclei (fine levels).

**Evidence:** Per-level diagnostic shows corrections of right magnitude (0.66x of |delta|) but wrong direction at most levels. This is the classic signature of missing information, not missing capacity.

### Root Cause #2: Multi-Task Interference (Training) — RESOLVED

The Kendall uncertainty weighting actively suppressed rho_s learning. Single-task ablation improved train from 2.99x to 1.28x.

**Status:** Resolved by switching to single-task mode.

### Root Cause #3: Level Sample Imbalance (Data)

With only 13 training molecules:
- Levels 11-12: ~37k samples -> 0.99x ratio (BEATS baseline)
- Levels 0-7: <500 samples each -> >1x (adds noise)
- Levels 15-17: <100 samples -> >1x (adds noise)

The model learns useful corrections where it has enough data. Noise at data-starved levels dominates the aggregate.

**Evidence:** Level masking simulation — even zeroing predictions at under-represented levels gives 1.006x (still 0.6% above baseline).

---

## Path Forward

### Step 1: Option A — More Training Data + Level Masking (~1-2 sessions)

**Goal:** Address root cause #3 (data starvation at extreme levels).

**Actions:**
1. Generate training data for additional W4-11 molecules (N2O, H2S, OCS, F2, Cl2, HOCl, SO2, etc.) using existing `dump_training_functions` pipeline
2. Rebuild `training_dataset.h5` with expanded molecule set
3. Add level-aware loss weighting to `SingleTaskLoss`: zero or downweight gradients at levels with <N samples (N to be tuned)
4. Retrain single-task model
5. Re-run `compare_densities.py` to check if ratio drops below 1.0

**Success criteria:** Per-leaf MSE ratio < 1.0 on training data, ratio < 1.0 on held-out validation.

**No C++ changes needed. No Adrian dependency.**

### Step 2: Option B — Parent Node Features (~3-4 sessions)

**Goal:** Address root cause #1 (insufficient multi-scale context).

**Actions:**
1. Modify `dataset_builder.py` to extract parent node s-coefficients from pymra `FunctionTree` (parent key = Key(n-1, l//2) — already available in the tree structure, no C++ changes needed)
2. Add parent feature channels to HDF5 dataset
3. Modify `HaloEncoder` or add `ParentEncoder` in `model.py` to consume parent features
4. Retrain and evaluate

**Success criteria:** Per-leaf MSE ratio < 0.9 on training data, < 1.0 on validation. Compare densities ratio < 1.0.

**No C++ changes needed. No Adrian dependency.**

### Step 3: Re-run SCF Test

After Option A (or A+B), re-run the SCF convergence test:
- Use `mra_nn/slurm/scf_test.sh` (already built and tested)
- Use `mra_nn/compare_densities.py` for density comparison
- Compare iteration counts, final energies, dipole moments

**Success criteria:** ML-guess SCF converges to the same ground state as baseline, in fewer iterations.

---

## Key Files

| File | Purpose |
|------|---------|
| `mra_nn/compare_densities.py` | L2 norm, per-leaf MSE, dipole comparison between densities |
| `mra_nn/predict.py` | ML inference (single-task and multi-task modes) |
| `mra_nn/model.py` | MRANet architecture |
| `mra_nn/losses.py` | SingleTaskLoss and UncertaintyWeightedLoss |
| `mra_nn/train.py` | Training loop |
| `mra_nn/dataset_builder.py` | HDF5 dataset construction from pymra trees |
| `mra_nn/configs/single_task.yaml` | Current training config |
| `mra_nn/slurm/scf_test.sh` | End-to-end SCF convergence test |
| `mra_nn/slurm/build_moldft_hdf5.sh` | Build script bypassing cmake |
| `src/madness/chem/SCF.cc` | ML density injection point (after line 1032) |
| `src/apps/molresponse/tools/h5_to_archive.cpp` | HDF5 to MADNESS archive converter |
| `mra_nn/diagnose_head.py` | Per-level head output diagnostic (uncommitted) |
| `mra_nn/estimate_masked.py` | Level masking simulation (uncommitted) |
| `docs/superpowers/plans/2026-07-31-mra-nn-single-task-ablation.md` | Single-task ablation plan and results |

---

## Option B Decisions (2026-08-10)

### Decision 17: Inference-Time Level Clamping (Step A)
- **When:** 2026-08-10 (commit `3db3b6d`)
- **What:** Added `--use-model-levels` to predict.py. At levels outside the set, output rho0_s unchanged. Default: model at levels 10-14 only.
- **Why:** Diagnostic showed model beats baseline at 10-14 but produces garbage at level 0 and adds noise at 1-9 and 15-17. Clamping isolates the useful predictions.
- **Result:** SCF test (job 2116753) — correct electronic state recovered (energy -114.85038032 vs baseline -114.85038034), but same iteration count (12). Fine-level accuracy alone does not drive SCF convergence.
- **Verdict:** Confirmed the hypothesis: coarse levels determine SCF iterations. Model needs multi-scale context to help there.

### Decision 18: Parent Feature Integration — Concatenation
- **When:** 2026-08-10
- **What:** Add parent node rho0/vnuc s-coefficients (1024 floats) by concatenating to the trunk input. No separate encoder.
- **Why:** Simplest approach that provides cross-level context. Trunk widens from 1792 to 2816 (+1M params in first layer). FiLM conditioning already adapts per-level, so no architectural novelty needed.
- **Alternatives rejected:**
  - Separate parent encoder MLP (adds ~400K params and a new module to tune — premature complexity)
  - FiLM conditioning from parent (conflates two conditioning signals — level and parent content — making it harder to diagnose failures)
- **Verdict:** Pending training results.

### Decision 19: Level 0 Zero-Padding for Missing Parent
- **When:** 2026-08-10
- **What:** At level 0 (root node), parent_rho0_s and parent_vnuc_s are zero-padded since no parent exists (n-1 = -1).
- **Why:** Level 0 has only 3 val samples and is already masked during training. The FiLM level embedding gives level 0 its own learned representation, so the model can learn that zeros mean "no parent." Alternatives (skip level 0, sentinel value) add complexity for 0.03% of data.
- **Verdict:** Pending training results.

### Decision 20: Train from Scratch (No Transfer)
- **When:** 2026-08-10
- **What:** Option B model trains from random init, not from Option A checkpoint.
- **Why:** First trunk layer shape changes (1792 -> 2816). Could theoretically transfer all layers except the first, but the first layer is where the parent signal enters — it's the most important layer to learn fresh. Transfer of later layers from a model that never saw parent features is unlikely to help and adds complexity.
- **Verdict:** Pending training results.

### Decision 21: Input Features — rho0 and vnuc Parents Only (No rho Target Parent)
- **When:** 2026-08-10
- **What:** Parent features are extracted from rho0 and vnuc trees only. Not from the converged density (rho) tree.
- **Why:** At inference time, only rho0 and vnuc are available — rho is what we're trying to predict. Including rho parent features in training would create a data leak.
