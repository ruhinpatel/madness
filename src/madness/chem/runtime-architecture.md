# MADNESS Chemistry Architecture

## Overview
- Molecular model: `Molecule` aggregates atoms, cores, and geometry; depends on tensors and world vectors (`src/madness/chem/molecule.h:129` with includes `tensor.h`, `world/vector.h`, `atomutil.h`, `corepotential.h`).
- SCF driver: `SCF` orchestrates HF/DFT and pulls in MRA/tensor/chem components (`src/madness/chem/SCF.h:190` after includes `mra/mra.h`, `tensor/solvers.h`, `tensor/distributed_matrix.h`, `molecularbasis.h`, `corepotential.h`, `xcfunctional.h`, `potentialmanager.h`). Typedefs bind MRA and tensor types to the solver (`Function<double,3>`, `SeparatedConvolution<double,3>`, `DistributedMatrix<double>` in SCF.h:36-55).
- Supporting chemistry libs: Basis/potential/functionals (`molecularbasis.h`, `corepotential.h`, `gth_pseudopotential.h`, `xcfunctional.h`) supply operators and grids consumed by SCF; functors/operators often rely on MRA Functions and tensor math (see their inclusion in SCF.h:36-55).
- Distributed math: Electronic structure matrices use `DistributedMatrix` (tensor/distributed_matrix.h:59-71,388) and solvers (`tensor/solvers.h`) within SCF.

## Diagram
Regenerate during docs build to avoid drift.
```mermaid
graph TD
  Molecule[molecule.h Molecule] --> SCF[SCF (SCF.h)]
  SCF --> MRA[mra/mra.h Functions/Operators]
  SCF --> Tensor[Tensor/DistributedMatrix (tensor.h/distributed_matrix.h)]
  SCF --> Basis[basis/potential/xc (chem headers)]
  Tensor --> World[World (world.h)] %% distribution for matrices
  MRA --> World %% distributed function trees
```

## How Data Moves
- Geometry/build: `Molecule` constructs atomic data using tensors/world vectors (molecule.h:129 and includes), feeding SCF setup.
- Function/potential setup: SCF includes MRA (`mra/mra.h`) and chem basis/potential headers; typedefs bind `Function` and `SeparatedConvolution` to 3D grids (SCF.h:36-55).
- Matrix assembly/solve: SCF uses `DistributedMatrix<double>` and solver utilities (`tensor/distributed_matrix.h:59-71,388`; `tensor/solvers.h`) to build and solve Fock/overlap systems across the `World`.
- Iteration: SCF logic (SCF.h:170-190) coordinates these components, invoking MRA operators and tensor solvers; world distribution ensures parallel execution.

## Index Table
- `molecule.h` (`Molecule`): builds geometry, uses tensor/world vector types (molecule.h:129).
- `SCF.h` (`SCF`): central driver referencing MRA/tensor/chem headers and typedefs for functions/operators/matrices (SCF.h:36-55,170-190).
- `molecularbasis.h`, `corepotential.h`, `gth_pseudopotential.h`, `xcfunctional.h`: provide basis/potential/functionals consumed by SCF.
- `tensor/distributed_matrix.h`, `tensor/solvers.h`: distributed matrices and solvers used in SCF.
- `mra/mra.h`: function/operator API used by SCF.
