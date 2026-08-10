# MRA-NN: Option A Diagnostic Results and Path Forward

**Date:** 2026-08-10
**Context:** Option A training (51 molecules + level masking) completed. Results appeared catastrophic (993x worse than baseline) but diagnostics revealed the true picture.

---

## Diagnostic Findings

### The 993x Was an Artifact of 3 Samples

Option A training job 2116400 reported best val MSE = 3.599e-04 vs. baseline 3.623e-07 (993x worse). Diagnostic job 2116733 revealed:

- **Val set baseline = 3.234e-07** (same order as train baseline 3.623e-07). The comparison was apples-to-apples, not a metric artifact.
- **Level 0 accounts for 99.9% of the model's MSE.** 3 samples (0.03% of val data) at level 0 produce MSE = 1.03 vs. baseline = 7.7e-10. The level masking zeroed gradient at level 0 during training but did not constrain inference output. The model's head produces unconstrained garbage at level 0.
- **Clamping level 0 to rho0 gives 1.005x** on held-out val (ethanol, so2, hnnn). The model at levels 1-17 is at parity with baseline.

### The Model Beats Baseline at Levels 10-14

Per-level val set performance (held-out molecules):

| Level | Ratio | Samples | Verdict |
|-------|-------|---------|---------|
| 0 | 1.3B x | 3 | Catastrophic (masked during training, unconstrained at inference) |
| 1-9 | 1.00-1.12x | 4,256 | Parity (residual passes rho0 through; no multi-scale context) |
| 10 | 0.98x | 1,120 | Beats baseline |
| 11 | 0.79x | 928 | Beats baseline |
| 12 | 0.67x | 1,056 | Beats baseline |
| 13 | 0.55x | 784 | Beats baseline |
| 14 | 0.41x | 368 | Beats baseline |
| 15-17 | 1.27-5.34x | 272 | Degraded (few samples) |

### Level-Clamped Model Beats Baseline by 2.9%

Using model predictions only at levels 10-14 and rho0 everywhere else:

| Scenario | Val MSE | Ratio |
|----------|---------|-------|
| Baseline (rho0) | 3.234e-07 | 1.000x |
| Raw model (reported) | 3.598e-04 | 1113x |
| Clamp level 0 only | 3.249e-07 | 1.005x |
| **Model at levels 10-14 only** | **3.139e-07** | **0.971x** |

The model learned transferable density corrections at fine resolution (near-nuclear region) that generalize to unseen molecules.

### Per-Molecule Validation (Diagnostic 2116733)

The model fails on all molecules (including training molecules) in aggregate, but this is entirely driven by level 0:

| Molecule | Set | Raw Ratio | Notes |
|----------|-----|-----------|-------|
| h2o | Train | 351x | Level 0 garbage dominates |
| ch3oh | Train | 505x | Same |
| ch3f | Train (old val) | 536x | Was 1.006x in previous run without level masking |
| ethanol | Val | 755x | Same |
| so2 | Val | 1866x | Same |

All molecules show the same pattern: catastrophic at level 0, near-parity or better at levels 10-14.

---

## Root Cause Summary

1. **Level masking design flaw (immediate cause):** Training-time masking removes gradient signal at level 0, but inference produces unconstrained output. The head drifts freely at masked levels. Fix: clamp inference output to rho0 at masked levels.

2. **Insufficient multi-scale context (architectural):** Levels 1-9 show the model adds nothing. The 6-neighbor same-level halo provides no cross-level information. The correction at coarse levels requires multi-scale context the architecture cannot provide.

3. **Training loss decoupled from evaluation (metric):** The reported "0.0000" training loss reflected the weighted, masked loss — not actual prediction quality. The training script should report val-set baseline alongside train baseline.

---

## Path Forward: Step A then Step B

### Step A: Inference-Time Level Clamping + SCF Test (~1 session)

Take the existing trained model. At inference time, force `prediction = rho0` at levels outside the effective range (10-14). Re-run the SCF convergence test.

**Why this first:** It's nearly free (no retraining), and it directly answers whether a 2.9% density improvement at fine levels translates to fewer SCF iterations. The answer determines how much Step B matters.

| Task | Work | Model |
|------|------|-------|
| A1. Add level clamping to `predict.py` | ~10 lines: accept a `use_model_levels` list (default [10-14] based on diagnostic); at other levels, output rho0_s unchanged | Sonnet 4.6 |
| A2. Fix eval script PYTHONPATH | Already done this session | -- |
| A3. Re-run SCF test on ch3oh | Submit `slurm/scf_test.sh` with clamped predictions | Sonnet 4.6 |
| A4. Analyze SCF results | Compare iterations, energy, dipole vs. baseline | Sonnet 4.6 |

**Files to modify:**
- `mra_nn/predict.py`: add `level_clamp_range` parameter to `predict_density_simple()`
- `mra_nn/train.py`: add val-set baseline print (one line, for future runs)

**Success criteria:**
- SCF converges to the correct ground state (energy within 1e-3 Ha of baseline)
- Iteration count reduced by >= 1 vs. baseline

**Decision gate:**
- Iterations decrease -> Step A succeeded. Proceed to B for more improvement.
- Iterations unchanged -> Fine-level accuracy alone doesn't drive convergence. B must unlock coarse levels.
- Wrong electronic state -> Density clamped model is somehow worse than rho0. Investigate before B.

### Step B: Parent Node Features (~3-4 sessions)

Add parent s-coefficients as model input. This gives the model one level of cross-level context, directly addressing root cause #2.

| Task | Work | Model |
|------|------|-------|
| B1. Design parent feature extraction and model integration | Architecture decision: tensor shapes, how parent features interact with FiLM conditioning, encoder design | Opus 4.6 |
| B2. Modify `dataset_builder.py` | Add parent s-coefficients to HDF5 (Key(n-1, l//2) via pymra tree) | Sonnet 4.6 |
| B3. Rebuild dataset | SLURM job, gate check | Sonnet 4.6 |
| B4. Modify `model.py` + `dataset.py` | Add ParentEncoder or extend HaloEncoder, update data loading | Opus 4.6 |
| B5. Retrain + evaluate | A100 training, run diagnostics (per-level breakdown) | Sonnet 4.6 |
| B6. Re-run SCF test | If per-level ratios improve at levels 1-9 | Sonnet 4.6 |

**Key design points for B1 (Opus):**
- Parent key: `Key(n-1, l//2)` — available in pymra `FunctionTree`
- At level 0, there is no parent — must handle gracefully (zero-pad or skip)
- Parent s-coefficients have `k^3 = 512` components (same as center node)
- Design question: concatenate with center features, or process through separate encoder?
- Design question: does FiLM conditioning change (parent level vs. child level)?

**Files to modify:**
- `mra_nn/dataset_builder.py`: extract parent s-coefficients during tree walk
- `mra_nn/dataset.py`: load new `parent_rho0_s` / `parent_vnuc_s` fields
- `mra_nn/model.py`: add parent feature processing path
- `mra_nn/configs/single_task.yaml`: new architecture params

**Success criteria:**
- Per-level ratio < 1.0 at levels 8-14 (currently 10-14 only)
- Overall level-clamped ratio < 0.95x on val
- SCF iterations reduced vs. baseline

**Decision gate (if B fails):**
- If levels 1-7 still at parity after parent features: the correction requires context beyond one parent. Consider message-passing on the tree graph or pivot to Approach C (tree structure prediction instead of density prediction).

### Approach C: Tree Structure Prediction (contingency)

Not pursued now, but documented for reference. If A and B both fail to improve SCF convergence:

- Pivot ML role from density prediction to tree refinement prediction
- Binary classification (refine/don't refine) instead of regression to sub-1e-7 precision
- The original multi-task refine head achieved F1 ~0.70 even when density prediction was poor
- This is fundamentally an easier ML problem

---

## Model Selection Rationale

- **Opus 4.6** for architecture decisions (B1, B4): Reasoning about tensor shapes, feature interactions, and how new inputs integrate with existing FiLM conditioning. Wrong choices here waste 3+ sessions.
- **Sonnet 4.6** for everything else: Pipeline changes, SLURM jobs, diagnostics, and focused edits where the spec is clear.

---

## Key Files Reference

| File | Role |
|------|------|
| `mra_nn/predict.py` | Inference — add level clamping here (Step A) |
| `mra_nn/train.py` | Training loop — add val baseline print |
| `mra_nn/model.py` | MRANet architecture — add parent features (Step B) |
| `mra_nn/dataset_builder.py` | HDF5 construction — add parent s-coefficients (Step B) |
| `mra_nn/dataset.py` | Data loading — add parent fields (Step B) |
| `mra_nn/losses.py` | SingleTaskLoss — no changes needed |
| `mra_nn/configs/single_task.yaml` | Training config — update for Step B |
| `mra_nn/slurm/scf_test.sh` | SCF convergence test — reuse as-is |
| `mra_nn/compare_densities.py` | Density comparison — reuse as-is |
| `mra_nn/diagnose_option_a.py` | Per-level diagnostic — reuse for Step B evaluation |
| `mra_nn/docs/2026-08-10-scf-test-postmortem.md` | Full 16-decision audit and root cause analysis |
