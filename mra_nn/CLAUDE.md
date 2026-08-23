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

## Current Status (2026-08-22)

**Two tracks: Paper + Path 2 (Dalton density as ML input)**

All three ML approaches to SCF acceleration using MRANet hit the same architectural limit (Decisions 1-27). Direction decided 2026-08-22:

### Track 1: Negative Results Paper (JCTC)
- Design spec: `docs/superpowers/specs/2026-08-22-negative-results-paper-design.md`
- Implementation plan: `docs/superpowers/plans/2026-08-22-negative-results-paper.md`
- Paper drafts: `paper/` directory (sections/, figures/, tables/)
- Status: plan written, execution not started

### Track 2: Path 2 — Dalton Density as ML Input
- Use Dalton (Gaussian basis) density instead of rho0 → predict small Dalton→MRA residual
- Email sent to Adrian (2026-08-22) with technical questions about his Dalton→MADNESS pipeline
- Status: blocked on Adrian's reply

### Final Job Results (Decision 27)
- 2119939: refine-only training — COMPLETED, best val refine F1=0.8599, gate FAIL
- 2119940: SCF tree-walk — COMPLETED, 31 iter vs baseline 12 (architectural limit confirmed)

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

### Key Scripts
- `gen_training_data.sh` — all 51 molecules
- `gen_training_data_new.sh` — 35 new molecules only
- `build_dataset.sh` — runs dataset_builder.py
- `eval_option_a.sh` — predicts + compares densities on ch3oh and ethanol
- `slurm/train_single_task.sh` — GPU training
- `slurm/scf_test.sh` — end-to-end SCF convergence test

## Verification

**IMPORTANT:** Never claim a task is done without running the relevant check first.

- After generating .mad.h5 files → run `python validate_h5.py <mol_dir>`
- After adding a new molecule → confirm it's in both `molecules/` AND `MOLECULES` array in `gen_training_data.sh`
- After writing a new `.in` file → dry-run: `mpirun -np 1 dump_training_functions --input=<file>`
- After a Slurm job completes → grep `.err` for errors and `.out` for `converged`/`FAILED`
- After writing a new training config → run `python mra_nn/validate_config.py <config>` (from repo root, with mra_nn venv active) before submitting. This checks all required keys AND molecule availability. Do this before `/slurm-submit`.
- After writing a new training config → also trace every non-default flag through `train.py` and confirm the loss function matches the stated objective. Checkpoint/gate metric changes are not loss function changes. Read the code path, don't assume the flag name describes what it does.
