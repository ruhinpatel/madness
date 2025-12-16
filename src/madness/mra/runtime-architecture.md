# MADNESS MRA Architecture

## Overview
- Main entry: `mra.h` pulls in world, tensor, and core MRA headers and provides `startup(World&, ...)` to broadcast defaults and tables (src/madness/mra/mra.h:68-90). It reexports keys, two-scale data, Legendre tools, defaults, factory, and impl headers (mra.h:96-114).
- Function functor contracts: `FunctionFunctorInterface` and `FunctionInterface` define how analytic/composite sources feed Function trees (src/madness/mra/function_interface.h:76,317). Dependencies: tensor (`tensor.h`, `gentensor.h`) and geometry keys (`key.h`, `function_common_data.h`).
- Construction/defaults: `FunctionFactory` builds `Function<T,NDIM>` instances (src/madness/mra/function_factory.h:94) using boundary/accuracy defaults in `FunctionDefaults` (src/madness/mra/funcdefaults.h:106).
- Operators: Separated convolution/operator helpers produce `Function` results from kernels (src/madness/mra/operator.h:72-91); they rely on functor interfaces and tensor types.
- World touchpoints: Many APIs accept `World&` for distributed trees and archiving (e.g., `Function`/`FunctionImpl` accessors in src/madness/mra/mra.h:700,929,1540,2127), and parallel archives are pulled via `world/parallel_archive` (mra.h:98-104).

## Diagram
Regenerate during docs build to avoid drift.
```mermaid
graph TD
  MRA[mra.h startup] --> Functor[FunctionFunctorInterface (function_interface.h)]
  MRA --> Factory[FunctionFactory (function_factory.h)]
  MRA --> Defaults[FunctionDefaults (funcdefaults.h)]
  MRA --> Ops[Operators (operator.h)]
  Functor --> Funcs[Function/FunctionImpl (mra.h)]
  Factory --> Funcs
  Defaults --> Factory
  Ops --> Funcs
  Funcs --> World[World (world.h)] %% distribution/archives
```

## How Data Moves
- Initialization: `startup(World&,...)` broadcasts two-scale, quadrature, defaults across ranks (mra.h:68-90).
- Building functions: Clients supply `FunctionFunctorInterface` implementations (function_interface.h:76-136) to `FunctionFactory` (function_factory.h:94) which instantiates `Function<T,NDIM>` with defaults (`FunctionDefaults` in funcdefaults.h:106).
- Applying operators: Operator helpers (operator.h:72-91) take existing `Function` trees and produce new ones, using functor interfaces and tensor backends.
- Distributed execution: Function methods access `World&` for parallel ops and serialization (mra.h:700,929,1540,2127), leveraging world archives for I/O (mra.h:98-104).

## Index Table
- `mra.h`: exports startup and pulls core pieces (world/tensor/MRA) (mra.h:68-114).
- `function_interface.h`: declares `FunctionFunctorInterface`, `FunctionInterface` (function_interface.h:76,317).
- `function_factory.h`: declares `FunctionFactory` (function_factory.h:94).
- `funcdefaults.h`: declares `FunctionDefaults` (funcdefaults.h:106).
- `operator.h`: defines separated convolution helpers (operator.h:72-91).
