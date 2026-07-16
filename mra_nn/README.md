# MRA-NN: Neural Network for Electron Density Prediction in the Multiwavelet Basis

Train a neural network to predict electron density ρ(r) from the Coulomb potential V(r), both represented in the MRA (multiwavelet) basis. The goal is to provide a better SCF initial guess than the promolecular density (ρ₀), reducing convergence iterations in MADNESS.

## Directory Layout

```
mra_nn/
├── molecules/              # MADNESS input files (h2o, nh3, ch4, co2, hf)
├── training_data/<mol>/    # Generated HDF5 data per molecule
│   ├── rho0.mad.h5         #   Promolecular density (no SCF needed)
│   ├── vnuc.mad.h5         #   Nuclear Coulomb potential (no SCF needed)
│   └── rho.mad.h5          #   Converged SCF electron density
├── gen_training_data.sh    # Slurm pipeline: dump rho0/vnuc → moldft → dump rho
├── build_dump_training.sh  # Slurm script to compile the C++ tool
├── validate_h5.py          # Validates .mad.h5 structure (/meta, /keys, /coeffs)
├── logs/                   # Slurm build and run logs
├── step0/                  # Earlier smoke test (Step 0: H2O moldft + dump_mra_trees)
└── docs/plans/             # Implementation plans
```

## Dependencies

- **MADNESS** fork at `/gpfs/projects/rjh/ruhin/madness-ruhin/` (branch `feat/mra-nn-data`)
- **Build dir**: `/gpfs/projects/rjh/ruhin/madness-build-hdf5/`
- GCC 13.2.0, Intel OneAPI 2024.2 (MPI, MKL, TBB), HDF5 1.12.1
- Python 3 with `h5py` (for validation)
- **pymra** (downstream consumer): `/gpfs/projects/rjh/adrian/pymra`

## Building the C++ Tool

The `dump_training_functions` binary is built as part of MADNESS's molresponse app:

```bash
# Source lives at:
#   madness-ruhin/src/apps/molresponse/tools/dump_training_functions.cpp

# Build via Slurm (compiles on long-40core):
cd /gpfs/projects/rjh/ruhin/mra_nn
/cm/shared/apps/slurm/21.08.8/bin/sbatch build_dump_training.sh

# Binary lands at:
#   madness-build-hdf5/src/apps/molresponse/dump_training_functions
```

## Generating Training Data

### 1. Add molecule inputs

Create a MADNESS input file in `molecules/`. Example (`molecules/h2o.in`):

```
dft
  xc lda
  k 6
  maxiter 20
end

molecule
  units atomic
  O  0.000  0.000  0.2226
  H  0.000  1.4276 -0.8904
  H  0.000 -1.4276 -0.8904
end
```

### 2. Run the pipeline

```bash
cd /gpfs/projects/rjh/ruhin/mra_nn
/cm/shared/apps/slurm/21.08.8/bin/sbatch gen_training_data.sh
```

The script runs three steps per molecule:
- **Step A**: `dump_training_functions --input=mol.in` → `rho0.mad.h5`, `vnuc.mad.h5`
- **Step B**: `moldft` → converged SCF archive (`mad.restartdata`)
- **Step C**: `dump_training_functions --input=mol.in --archive=mad.restartdata` → `rho.mad.h5`

Output lands in `training_data/<mol>/`.

### 3. Validate

```bash
python3 validate_h5.py
```

Checks each `.mad.h5` for correct `/meta` attributes (schema, k, thresh, ndim, n_nodes, etc.), `/keys` dataset, and `/coeffs` dataset.

## HDF5 Schema

Each `.mad.h5` file follows schema v1 from `function_hdf5_io.hpp`:

| Group/Dataset | Contents |
|---------------|----------|
| `/meta` | Attributes: schema, k, thresh, ndim, n_nodes, n_coeff_nodes, cell, tree_state, etc. |
| `/keys` | `(n_nodes, 3+ndim)` int64 — tree node keys (level, translations) |
| `/coeffs` | `(n_coeff_nodes, k^ndim)` float64 — leaf scaling coefficients |

## Current Status

- 5 molecules generated and validated (H2O, NH3, CH4, CO2, HF) — all at k=6, thresh=1e-4, xc=lda
- Next: scale to 10-20 molecules, then hand off to pymra for dataset construction and model training

## Key References

- **Paper**: Gong et al., *"GED-CRN Breaks the Data Barrier"* (19-molecule electron density prediction)
- **Notion**: search "MRA-NN Project" for full meeting notes, task tracker, and architecture discussion
- **Cluster**: SeaWulf, `long-40core` partition (1 node, 40 CPUs, up to 7 days)

## Cluster Notes

- `MAD_NUM_THREADS` must always be set to `ntasks - 1`
- Intel `setvars.sh` must be sourced before `set -u` (has unbound vars)
- `mpirun` needs `/cm/shared/apps/slurm/21.08.8/bin` in PATH for srun
- HDF5 writer requires single MPI rank (`mpirun -np 1`)
- `ParallelInputArchive` appends `.00000` to archive paths — pass the prefix only
