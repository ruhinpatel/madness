# writecoeffs Examples and HDF5 Tutorial

This directory contains examples for writing MADNESS function data in different
formats (`.dat`, JSON, and HDF5).

## Files to look at

- `writecoeff.cc` / `writecoeff2.cc` / `writecoeff3.cc`
  - text-based coefficient I/O patterns.
- `writecoeff_json.cc`
  - JSON-oriented coefficient workflow.
- `writecoeff_hdf5.cc`
  - HDF5-enabled example. Writes a simple dataset (`norm2`) to `fun.h5`.
- `FunctionIO.h`
  - baseline helper utilities for serializing/reconstructing function data.
- `FunctionIOHDF5.h`
  - in-progress richer data model for HDF5-style workflows.

## Prerequisites

- A configured MADNESS build tree.
- HDF5 module or installation visible to CMake when using HDF5.

On SeaWulf, this is typically:

```bash
module avail hdf5
module load hdf5
```

If CMake still cannot find HDF5, pass one of:

- `-DHDF5_ROOT=/path/to/hdf5/prefix`
- `-DHDF5_DIR=/path/to/hdf5/cmake/config/dir`

## Configure with HDF5 enabled

From repo root:

```bash
cmake -S . -B build-hdf5 -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DENABLE_HDF5=ON
```

If your environment needs explicit BLAS/LAPACK/MPI flags, add your usual
MADNESS configure arguments.

## Build and run

Build only the writecoeff examples:

```bash
ninja -C build-hdf5 writecoeff writecoeff_hdf5
```

Run:

```bash
./build-hdf5/src/examples/writecoeffs/writecoeff
./build-hdf5/src/examples/writecoeffs/writecoeff_hdf5
```

Expected outputs from `writecoeff_hdf5`:

- `fun.dat` (text coefficient data)
- `fun.h5` (HDF5 file with dataset `norm2`)

Optional inspection tools:

```bash
h5ls -r fun.h5
h5dump -d /norm2 fun.h5
```

## Running tests

If `BUILD_TESTING=ON`, this directory now registers:

- serial test: `writecoeff_hdf5_serial`
- MPI test: `writecoeff_hdf5_mpi2`
- MPI + threads test: `writecoeff_hdf5_mpi2_threads2`

Run only these tests from the build directory:

```bash
ctest -R writecoeff_hdf5 -V
```

Run only MPI-flavored checks:

```bash
ctest -R writecoeff_hdf5_mpi -V
```

## Quick HDF5 learning path in this directory

1. Start from `writecoeff.cc` to understand the MADNESS function setup:
   - construct a function with `FunctionFactory`
   - call `truncate()` and `norm2()`
2. Open `writecoeff_hdf5.cc`:
   - see `write_norm_to_hdf5(...)`
   - note direct HDF5 C API calls: `H5Fcreate`, `H5Dcreate2`, `H5Dwrite`
3. Compare against `FunctionIO.h`:
   - identify where tree/leaf values are available for export.
4. Review `FunctionIOHDF5.h`:
   - this shows a richer container (`FunctionIOData`) for shape, cell, keys,
     values, and coordinates.

## Integrating HDF5 with MADNESS data (recommended next steps)

Use this incremental approach.

### Step 1: Keep rank-0 only output

Current examples already use rank-0 for file output, which is simplest for
correctness while iterating.

### Step 2: Define an HDF5 layout for function data

A practical layout:

- `/meta/ndim` (scalar)
- `/meta/k` (scalar)
- `/meta/cell` (shape `[ndim, 2]`)
- `/meta/num_leaf_nodes` (scalar)
- `/leaf/nl` (shape `[num_leaf_nodes, ndim + 1]`, level + translations)
- `/leaf/values` (shape `[num_leaf_nodes, npts_per_box]`)

This maps closely to fields already present in `FunctionIOData`.

### Step 3: Write leaf data arrays

From `FunctionIO` traversal results:

- flatten per-node values into contiguous arrays
- store with fixed-size dimensions for easier readback

### Step 4: Add readback and reconstruction

- load metadata and leaf arrays
- reconstruct `Function<T,NDIM>` similarly to `read_function(...)` patterns
- compare norms/errors against the source function

### Step 5: Move to parallel HDF5 only if needed

After serial/rank-0 flow is stable:

- use MPI-enabled HDF5
- coordinate collective writes by rank ownership

## Troubleshooting

- Error: `ENABLE_HDF5=ON but HDF5 was not found`
  - load HDF5 module and/or set `HDF5_ROOT`/`HDF5_DIR`.
- `writecoeff_hdf5` target missing
  - confirm `-DENABLE_HDF5=ON` at configure time.
- Configure fails before HDF5 checks (MPI/LAPACK issues)
  - use your normal SeaWulf module stack and MADNESS configure flags first.
