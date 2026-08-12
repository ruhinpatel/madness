/// \file h5_to_archive.cpp
/// \brief Convert a structured HDF5 function (.mad.h5) to a MADNESS binary archive.
///
/// Usage: mpirun -np 1 h5_to_archive input.mad.h5 output_prefix
///
/// Writes output_prefix.00000 (MADNESS binary archive format) that can be
/// loaded with ParallelInputArchive<BinaryFstreamInputArchive>.

#ifdef MADNESS_HAS_HDF5

#include "../solvers/function_hdf5_io.hpp"
#include <madness/mra/mra.h>
#include <madness/world/MADworld.h>
#include <string>

using namespace madness;

int main(int argc, char** argv) {
    World& world = initialize(argc, argv);
    startup(world, argc, argv, true);

    if (argc != 3) {
        if (world.rank() == 0) {
            print("Usage: h5_to_archive input.mad.h5 output_prefix");
            print("  Converts structured HDF5 to MADNESS binary archive.");
        }
        finalize();
        return 1;
    }

    const std::string h5_path = argv[1];
    const std::string archive_prefix = argv[2];

    if (world.rank() == 0)
        print("Reading", h5_path);

    auto f = molresponse_v3::load_function_hdf5<double, 3>(world, h5_path);

    if (world.rank() == 0) {
        double trace = f.trace();
        print("  trace =", trace);
        print("Writing", archive_prefix);
    }

    archive::ParallelOutputArchive<archive::BinaryFstreamOutputArchive>
        ar(world, archive_prefix.c_str(), 1);
    ar & f;

    world.gop.fence();
    if (world.rank() == 0)
        print("Done.");

    finalize();
    return 0;
}

#else
#include <cstdio>
int main() {
    fprintf(stderr, "h5_to_archive: built without MADNESS_HAS_HDF5\n");
    return 1;
}
#endif
