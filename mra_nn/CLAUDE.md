# MRA-NN Project

Train a neural network V(r) → ρ(r) in the MRA (multiwavelet) basis as a better SCF initial guess than promolecular density in MADNESS.

## Key Paths

- Training data: `training_data/<mol>/` — rho0.mad.h5, vnuc.mad.h5, rho.mad.h5
- Molecule inputs: `molecules/<mol>.in`
- Slurm pipeline: `gen_training_data.sh`
- Job logs: `logs/gen_training_data_<jobid>.out/err`
- pymra (downstream): `/gpfs/projects/rjh/adrian/pymra`

## MADNESS Build

Use `madness-build-hdf5/` for all MRA-NN tools:
- `moldft` — SCF convergence
- `dump_mra_trees` — orbital export (Step 0)
- `dump_training_functions` — training data export (rho0/vnuc/rho)

## Molecule Inputs Standard

`xc lda`, `thresh 1e-6`, `k 8`, `maxiter 20`, units angstrom.

**IMPORTANT:** `k 8` must be set explicitly — `dump_training_functions` does not auto-derive k
from thresh. Omitting `k` causes k=-1 and a tensor assertion crash.

**Geometry source:** Coordinates are taken from the W4-11 thermochemical benchmark set
(Karton et al.) at `/gpfs/projects/rjh/ruhin/perf_pipeline/molecules/W4-11/`,
converted from bohr to angstrom. These are CCSD(T)/cc-pVTZ optimized geometries.
Note: ch3oh and h2o2 use the W4-11 entries `methanol` and `hooh` respectively.

**IMPORTANT:** New molecules must be added to both `molecules/` and the `MOLECULES` array in `gen_training_data.sh`.

## Current Status (2026-08-11)

**Approach 2: Tree Structure Prediction (refine-only training)**

### Molecules (51 total)
- **Original 16:** h2o, nh3, ch4, co2, hf, n2, co, hcn, c2h2, c2h4, c2h6, h2co, ch3oh, h2o2, hcl, ch3f
- **New 35 (Option A):** f2, cl2, clf, h2s, cs2, hnc, hof, hocl, n2o, ocs, so2, so3, f2o, cl2o, clcn, hno, hcno, hnco, hnnn, hocn, hcof, c-n2h2, t-n2h2, ch2f2, cf4, nh2cl, n2h4, ch3nh2, ketene, formic, acetaldehyde, ethanol, allene, oxirane, glyoxal

### Train/Val/Test Split
- **Train (45):** all except val/test below
- **Val (3):** ethanol, so2, hnnn
- **Test (3):** h2o2, c2h2, glyoxal

### Level Masking (in SingleTaskLoss)
- Hard cutoff: levels with <200 training samples → zero gradient
- Soft weighting: remaining levels weighted by sqrt(count / max_count)
- Configured via `min_level_samples` in `loss` section of config YAML

### Job Chain (2026-08-11)
- 2116969: refine_focused multi-task training — COMPLETED, refine F1=0.8568, gate FAIL (multi-task interference)
- 2116970: post-training SCF tree-walk — COMPLETED, 31 iter vs baseline 12 (expected given F1)
- **2117581: refine-only training (Decision 26) — PENDING** (pure FocalLoss, no density heads)

### Previous Job Chain (2026-08-10)
- 2116392: data generation for 35 new molecules — COMPLETED
- 2116396: dataset_builder.py (51 molecules → 24 GB training_dataset.h5) — COMPLETED, gate PASS
- 2116400: training on a100-long — COMPLETED
- 2116401: evaluation (ch3oh + ethanol density comparison) — COMPLETED
- 2116753: SCF test with level-clamped model (levels 10-14) — COMPLETED, neutral outcome

### Key Scripts
- `gen_training_data.sh` — all 51 molecules
- `gen_training_data_new.sh` — 35 new molecules only
- `build_dataset.sh` — runs dataset_builder.py
- `eval_option_a.sh` — predicts + compares densities on ch3oh and ethanol
- `slurm/train_single_task.sh` — GPU training
- `slurm/scf_test.sh` — end-to-end SCF convergence test

### Step A Results (2026-08-10)

**Level clamping diagnostic:** Option A training (job 2116400) appeared 993x worse than baseline, but diagnostic analysis revealed level 0 (3 samples, 0.03% of val data) accounted for 99.9% of MSE due to unconstrained inference at training-masked levels. Model beats baseline at levels 10-14 (0.41-0.98x on held-out val). Level-clamped model (use model only at levels 10-14, rho0 elsewhere) gives 0.971x overall.

**SCF test (job 2116753):** Level-clamped ML density vs rho0 on ch3oh.

| Metric | Baseline (rho0) | ML clamped [10-14] |
|--------|-----------------|---------------------|
| Final energy (Ha) | -114.85038034 | -114.85038032 |
| Total iterations | 12 | 12 |
| Dipole (a.u.) | 0.6448 | 0.6448 |

Correct electronic state recovered (previous unclamped test converged to wrong state). Iteration count unchanged — fine-level density improvement does not drive SCF convergence.

**Conclusion:** Coarse levels (1-9) determine SCF iteration count. Model adds nothing at those levels due to insufficient multi-scale context. Proceed to Option B (parent node features).

### Approach 2 History

**Density prediction exhausted** (Options A/B failed, Approach 3 confirmed 14/14 zero iteration savings). Pivoted to refinement classification — predict tree shape instead of density coefficients.

- **Decision 25:** Approach 2 design — binary refinement classification
- **Decision 26:** First attempt used `refine_focused=true` but loss was unchanged (multi-task interference). Fixed with `RefineOnlyLoss` — pure focal loss, no density heads. Job 2117581 pending.
- Full audit: `docs/2026-08-10-scf-test-postmortem.md` (Decisions 17-26)

## Verification

**IMPORTANT:** Never claim a task is done without running the relevant check first.

- After generating .mad.h5 files → run `python validate_h5.py <mol_dir>`
- After adding a new molecule → confirm it's in both `molecules/` AND `MOLECULES` array in `gen_training_data.sh`
- After writing a new `.in` file → dry-run: `mpirun -np 1 dump_training_functions --input=<file>`
- After a Slurm job completes → grep `.err` for errors and `.out` for `converged`/`FAILED`
- After writing a new training config → trace every non-default flag through `train.py` and confirm the loss function matches the stated objective. Checkpoint/gate metric changes are not loss function changes. Read the code path, don't assume the flag name describes what it does.
