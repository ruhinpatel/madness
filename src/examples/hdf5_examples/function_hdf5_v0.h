#pragma once

#include <madness/mra/mra.h>
#include <string>

namespace madness {
namespace io {
namespace hdf5 {

template <typename T>
void save_function_v0(
    const Function<T, 2>& f,
    const std::string& filename
);

template <typename T>
Function<T, 2> load_function_v0(
    World& world,
    const std::string& filename
);

}  // namespace hdf5
}  // namespace io
}  // namespace madness
