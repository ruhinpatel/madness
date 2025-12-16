# MADNESS Tensor Architecture

## Overview
- Core container: `Tensor` derives from `BaseTensor` and provides multidimensional dense storage (src/madness/tensor/tensor.h:317). `tensor.h` pulls alignment, BLAS wrappers, iterators (`aligned.h`, `mxm.h`, `tensorexcept.h`, `tensoriter.h`) and uses world memory/archive helpers (`world/posixmem.h`, `world/archive.h` in tensor.h:20-34).
- Base layer: `BaseTensor` sets layout traits, includes macros and slicing/types (`tensor_macros.h`, `type_data.h`, `slice.h`, `vector_factory.h`) (basetensor.h:24-46). Optional atomic instance counting uses `world/atomicint.h` (basetensor.h:40-45).
- Distributed matrices: `DistributedMatrixDistribution` and `DistributedMatrix` manage tiled layouts across a `World` communicator (distributed_matrix.h:59-71,388). Factory helpers like `column_distributed_matrix_distribution(World&,...)` tie distribution to world ranks.
- Linear algebra helpers: GMRES/solvers on tensors and distributed matrices (`gmres.h`, `solvers.h`) rely on tensor BLAS/LAPACK wrappers (`mxm.h`, `linalg_wrappers.h`, `tensor_lapack.h`).

## Diagram
Regenerate during docs build to avoid drift.
```mermaid
graph TD
  Base[BaseTensor (basetensor.h)] --> Tensor[Tensor (tensor.h)]
  Tensor --> BLAS[BLAS/LAPACK helpers (mxm.h, tensor_lapack.h)]
  Tensor --> DistMat[DistributedMatrix (distributed_matrix.h)]
  DistMat --> World[World (world.h)] %% distribution helpers
  BLAS --> Solvers[Solvers/GMRES (solvers.h/gmres.h)]
  DistMat --> Solvers
```

## How Data Moves
- Allocation/storage: `Tensor` instances allocate data (tensor.h:317) with layout/slice info from `BaseTensor` (basetensor.h:24-46).
- Local ops: BLAS/LAPACK wrappers (`mxm.h`, `tensor_lapack.h`) operate on Tensor views; exception handling via `tensorexcept.h`.
- Distributed layouts: `DistributedMatrixDistribution` constructors taking `World&` define tiling and ownership (distributed_matrix.h:59-71); `DistributedMatrix` inherits distribution and stores blocks (distributed_matrix.h:388).
- Solving: Iterative solvers (`gmres.h`, `solvers.h`) accept `DistributedMatrix`/Tensor inputs, chaining BLAS ops and (optionally) world-aware distributions.

## Index Table
- `tensor.h`: declares `Tensor` and includes base/iterators/BLAS (tensor.h:20-60,317).
- `basetensor.h`: declares `BaseTensor` and base macros/slices (basetensor.h:24-46).
- `distributed_matrix.h`: declares distribution helpers and `DistributedMatrix` using `World` (distributed_matrix.h:59-71,388).
- `mxm.h`, `tensor_lapack.h`, `linalg_wrappers.h`: BLAS/LAPACK integration for tensors.
- `gmres.h`, `solvers.h`: solver fronts built atop tensors/distributed matrices.
