# MRA-NN Training Data Generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate HDF5 training data (ρ₀, V_nuc, ρ) for 5 molecules on SeaWulf, producing `.mad.h5` files readable by `pymra` for MRA-NN training.

**Architecture:** A new C++ tool `dump_training_functions` is added to the MADNESS molresponse app. It uses the MADNESS `SCF` class to compute promolecular density (ρ₀) and nuclear potential (V_nuc) without running SCF, and optionally loads a converged density (ρ) from a restart archive. All functions are written to HDF5 using the existing `function_hdf5_io.hpp` schema (v1: /meta, /keys, /coeffs). A Slurm pipeline runs the tool on 5 molecules and a Python script validates the output.

**Tech Stack:** C++17, MADNESS MRA framework, Intel OneAPI MPI/MKL/TBB, HDF5 1.12.1, Python/h5py, Slurm on SeaWulf

## Global Constraints

- MADNESS fork: `/gpfs/projects/rjh/ruhin/madness-ruhin/`, branch `feat/mra-nn-data`
- Build dir: `/gpfs/projects/rjh/ruhin/madness-build-hdf5/`
- Project dir: `/gpfs/projects/rjh/ruhin/mra_nn/`
- k=6, thresh=1e-4, xc=lda for all molecules
- HDF5 writer requires NP=1 (single MPI rank) — `dump_training_functions` always runs with `mpirun -np 1`
- `MAD_NUM_THREADS = ntasks - 1` always
- Never add Co-Authored-By lines to commits
- Best model for C++ tasks: Opus 4.6 | For scripts/validation: Sonnet 4.6

---

### Task 1: Fix compile errors in dump_training_functions.cpp

**Model:** Opus 4.6

**Files:**
- Modify: `src/apps/molresponse/tools/dump_training_functions.cpp`

**Background:** Two compile errors from the last build attempt:
1. `load_molecule_near()` calls `Molecule(j["molecule"])` — `Molecule` has no JSON constructor. Fix: delete the function entirely; the molecule is already available as `calc.molecule` (loaded from `--input`).
2. `real_functor_3d` is not a recognized type at that call site. Fix: use `std::make_shared<MolecularGuessDensityFunctor>(...)` which returns `std::shared_ptr<FunctionFunctorInterface<double,3>>` — the type `.functor()` expects.

- [ ] **Step 1: Remove `load_molecule_near` and fix the archive block**

In `dump_training_functions.cpp`, delete the entire `load_molecule_near()` function (lines 63–78 approx) and update the archive block to use `calc.molecule` directly:

```cpp
// REMOVE this entire function:
// Molecule load_molecule_near(const std::string& archive_path) { ... }

// In the archive block, REPLACE:
//   Molecule mol2 = load_molecule_near(archive_path);
//   auto gs = GroundState::from_archive(world, archive_path, mol2);
// WITH:
auto gs = GroundState::from_archive(world, archive_path, calc.molecule);
```

- [ ] **Step 2: Fix MolecularGuessDensityFunctor instantiation**

```cpp
// REPLACE:
real_functor_3d rho0_functor(
    new MolecularGuessDensityFunctor(calc.molecule, calc.aobasis));
real_function_3d rho0 =
    real_factory_3d(world)
        .functor(rho0_functor)
        .truncate_on_project();

// WITH:
auto rho0_functor = std::make_shared<MolecularGuessDensityFunctor>(
    calc.molecule, calc.aobasis);
real_function_3d rho0 =
    real_factory_3d(world)
        .functor(rho0_functor)
        .truncate_on_project();
```

- [ ] **Step 3: Remove unused includes**

Remove `#include <fstream>` and `#include <filesystem>` from the anonymous namespace area if they were added there (they are already present via GroundState.hpp transitively). Leave the top-level includes untouched.

- [ ] **Step 4: Submit build job**

```bash
/cm/shared/apps/slurm/21.08.8/bin/sbatch /gpfs/projects/rjh/ruhin/mra_nn/build_dump_training.sh
```

Expected: `Submitted batch job XXXXXXX`

- [ ] **Step 5: Poll until complete**

```bash
/cm/shared/apps/slurm/21.08.8/bin/sacct -j <JOBID> --format=JobID,State,ExitCode,Elapsed
```

Expected: State=COMPLETED, ExitCode=0:0

- [ ] **Step 6: Verify binary exists**

```bash
ls -lh /gpfs/projects/rjh/ruhin/madness-build-hdf5/src/apps/molresponse/dump_training_functions
```

Expected: file ~5–20 MB

**Gate:** Binary exists at the path above with exit code 0.

---

### Task 2: Generate ρ₀ + V_nuc for all 5 molecules (no SCF)

**Model:** Sonnet 4.6

**Files:**
- Run: `mra_nn/gen_training_data.sh` (Steps A only, or full pipeline)
- Output: `mra_nn/training_data/<mol>/rho0.mad.h5`, `vnuc.mad.h5`

**Molecules:** h2o, nh3, ch4, co2, hf

- [ ] **Step 1: Submit the full data generation job**

```bash
/cm/shared/apps/slurm/21.08.8/bin/sbatch /gpfs/projects/rjh/ruhin/mra_nn/gen_training_data.sh
```

Note: `gen_training_data.sh` runs Steps A (rho0+vnuc), B (moldft SCF), and C (converged rho) sequentially for all 5 molecules. The HBM partition requires ≥6 nodes — verify the script uses `hbm-medium-96core` with `--nodes=6` minimum, or switch to `long-40core` (1 node allowed).

- [ ] **Step 2: Poll until complete**

```bash
/cm/shared/apps/slurm/21.08.8/bin/sacct -j <JOBID> --format=JobID,State,ExitCode,Elapsed
```

- [ ] **Step 3: Check output files**

```bash
ls -lh /gpfs/projects/rjh/ruhin/mra_nn/training_data/*/rho0.mad.h5
ls -lh /gpfs/projects/rjh/ruhin/mra_nn/training_data/*/vnuc.mad.h5
ls -lh /gpfs/projects/rjh/ruhin/mra_nn/training_data/*/rho.mad.h5
```

Expected: 10 files (rho0 + vnuc) minimum, 15 if converged rho also succeeded.

**Gate:** Both `rho0.mad.h5` and `vnuc.mad.h5` exist and are >0 bytes for all 5 molecules.

---

### Task 3: Validate all H5 files

**Model:** Sonnet 4.6

**Files:**
- Create: `mra_nn/validate_h5.py`

- [ ] **Step 1: Write validation script**

```python
#!/usr/bin/env python3
"""Validate MRA-NN training HDF5 files: schema + electron count check."""
import h5py
import sys

BASE = "/gpfs/projects/rjh/ruhin/mra_nn/training_data"

# Expected electron counts per molecule
MOLECULES = {
    "h2o": 10,
    "nh3": 10,
    "ch4": 10,
    "co2": 22,
    "hf":  10,
}

errors = []
for mol, nelec in MOLECULES.items():
    for fname in ["rho0.mad.h5", "vnuc.mad.h5", "rho.mad.h5"]:
        path = f"{BASE}/{mol}/{fname}"
        try:
            with h5py.File(path, "r") as f:
                keys = set(f.keys())
                missing = {"meta", "keys", "coeffs"} - keys
                if missing:
                    errors.append(f"FAIL {path}: missing groups {missing}")
                else:
                    n_nodes = f["meta"]["n_nodes"][()]
                    n_coeff = f["meta"]["n_coeff_nodes"][()]
                    print(f"  OK  {mol}/{fname}: {n_nodes} nodes, {n_coeff} coeff nodes")
        except FileNotFoundError:
            if fname == "rho.mad.h5":
                print(f"  SKIP {mol}/{fname}: not yet generated (needs moldft)")
            else:
                errors.append(f"FAIL {path}: file not found")

if errors:
    print("\nErrors:")
    for e in errors:
        print(" ", e)
    sys.exit(1)
else:
    print("\nAll files valid.")
```

- [ ] **Step 2: Run validation**

```bash
python3 /gpfs/projects/rjh/ruhin/mra_nn/validate_h5.py
```

Expected output: `OK` lines for each file, `All files valid.` at end.

**Gate:** Script exits 0 with no FAIL lines.

---

## Fix needed in gen_training_data.sh before Task 2

The data generation script currently requests `hbm-medium-96core` with 1 node, but that partition requires ≥6 nodes. Before submitting Task 2, update the script:

```bash
#SBATCH --partition=long-40core
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --time=12:00:00
```

And update `MAD_NUM_THREADS=39` and use `long-40core` (allows 1 node, 7-day limit).
