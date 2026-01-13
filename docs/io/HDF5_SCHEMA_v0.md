# HDF5 Schema v0 (Phase 0)

## Purpose and scope
This document defines the schema contract for Phase 0 (v0) HDF5 support for MADNESS.
Scope is strictly limited to exactly one function per file and a single 2D real-valued MADNESS function.

## Conceptual model
The function is represented by leaf-wise values derived from wavelet coefficients.
The adaptive tree structure and raw coefficients are not stored.
For each leaf node, the file stores:
- refinement level
- translation[2]
- values[k*k] obtained via `coeffs2values`
Reconstruction uses `values2coeffs` and accumulation into a function.

## File layout
```
/
+-- metadata/
¦   +-- schema_version        ("v0")
¦   +-- dimension             (int, value = 2)
¦   +-- value_type            ("real")
¦   +-- k                     (wavelet order)
¦   +-- cell                  (double[2][2])
¦   +-- num_leaf_nodes
¦   +-- optional description / timestamp
¦
+-- function/
    +-- leaf_level            (int[N])
    +-- leaf_translation      (int[N][2])
    +-- leaf_values           (double[N][k*k])
```

## Reconstruction semantics
- Read metadata.
- Assert `schema_version == "v0"` and `dimension == 2`.
- Set MADNESS defaults (`FunctionDefaults<2>::set_cell`) using `cell` and `k` from metadata.
- Create an empty function with `FunctionFactory`.
- For each leaf node:
  - Build `Key<2>(level, translation)`.
  - Convert `values` to coefficients using `values2coeffs`.
  - Accumulate the coefficients into the function.
- Call `verify_tree()` after reconstruction.

## Precision and correctness
Values are stored in double precision.
Bitwise reproducibility is not guaranteed.
Correctness is defined by numerical agreement within tolerance.

## Non-goals
- No 3D support.
- No vector-of-functions.
- No restart bundles.
- No parallel HDF5 I/O.
- No native coefficient storage.

## Minimal example description
A single file contains metadata for a 2D real-valued function with wavelet order `k`, a 2x2 `cell`, and `num_leaf_nodes`.
The function section stores one row per leaf with its level, 2D translation, and `k*k` values derived from `coeffs2values`.
This data is sufficient to reconstruct the function within numerical tolerance using the specified semantics.