#pragma once

/// \file function_hdf5_v1.h
/// \brief General-NDIM HDF5 serialization for MADNESS Function objects.
///
/// Extends v0 (hardcoded to NDIM=2) to arbitrary dimension.
/// On-disk layout:
///
///   /metadata
///       schema_version   string    "v1"
///       dimension        int       NDIM
///       value_type       string    "real"  (complex: future work)
///       k                int       wavelet order
///       cell             [NDIM][2] double  simulation box lo/hi per dimension
///       num_leaf_nodes   hsize_t
///
///   /function
///       leaf_level       [N]         int     refinement level of each leaf
///       leaf_translation [N][NDIM]   int     box index per spatial dimension
///       leaf_values      [N][k^NDIM] double  real-space values at each leaf

#include <madness/mra/mra.h>
#include <string>
#include <cstddef>

namespace madness {
namespace io {
namespace hdf5 {

/// Save a MADNESS function to HDF5.
///
/// Calls f.reconstruct() internally. Only rank-0 writes; all ranks
/// participate in the final fence.
///
/// Explicit instantiations: double/float x NDIM = 1, 2, 3, 6
template <typename T, std::size_t NDIM>
void save_function_v1(
    const Function<T, NDIM>& f,
    const std::string& filename
);

/// Load a MADNESS function from HDF5 written by save_function_v1.
///
/// Only rank-0 reads; data is broadcast to all ranks before the function
/// tree is distributed according to the current process map.
///
/// Explicit instantiations: double/float x NDIM = 1, 2, 3, 6
template <typename T, std::size_t NDIM>
Function<T, NDIM> load_function_v1(
    World& world,
    const std::string& filename
);

}  // namespace hdf5
}  // namespace io
}  // namespace madness
