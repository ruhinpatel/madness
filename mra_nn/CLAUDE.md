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

## Current Status (2026-07-30)

- Training data regeneration at k=8/thresh=1e-6 in progress (Slurm job 2106527)
- All Step 6 model code complete on feat/mra-nn-data
- Awaiting k=8 data to run dataset_builder.py and GPU training (Task 7)

## Verification

**IMPORTANT:** Never claim a task is done without running the relevant check first.

- After generating .mad.h5 files → run `python validate_h5.py <mol_dir>`
- After adding a new molecule → confirm it's in both `molecules/` AND `MOLECULES` array in `gen_training_data.sh`
- After writing a new `.in` file → dry-run: `mpirun -np 1 dump_training_functions --input=<file>`
- After a Slurm job completes → grep `.err` for errors and `.out` for `converged`/`FAILED`
