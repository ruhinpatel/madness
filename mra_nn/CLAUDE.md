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

`xc lda`, `thresh 1e-4`, `k 6`, `maxiter 20`, units angstrom.

**IMPORTANT:** New molecules must be added to both `molecules/` and the `MOLECULES` array in `gen_training_data.sh`.

## Current Status (2026-07-08)

- Step 0 done: 5 molecules (h2o, nh3, ch4, co2, hf), all 15 .mad.h5 files validated
- feat/mra-nn-data pushed to GitHub (ruhinpatel/madness)
- Step 1 in progress: run `scripts/validate_vs_cube.py` against mo_0.mad.h5

## Verification

**IMPORTANT:** Never claim a task is done without running the relevant check first.

- After generating .mad.h5 files → run `python validate_h5.py <mol_dir>`
- After adding a new molecule → confirm it's in both `molecules/` AND `MOLECULES` array in `gen_training_data.sh`
- After writing a new `.in` file → dry-run: `mpirun -np 1 dump_training_functions --input=<file>`
- After a Slurm job completes → grep `.err` for errors and `.out` for `converged`/`FAILED`
