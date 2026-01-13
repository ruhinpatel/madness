#include <madness/mra/mra.h>
#include <madness/io/hdf5/function_hdf5_v0.h>

#include <cmath>
#include <cassert>

using namespace madness;

static const int kDim = 2;
using coordT = Vector<double, kDim>;

static double gaussian(const coordT& r) {
    const double x = r[0];
    const double y = r[1];
    return std::exp(-(x * x + y * y));
}

int main(int argc, char** argv) {
    World& world = initialize(argc, argv);
    startup(world, argc, argv);

    const double L = 4.0;
    const long k = 5;
    const double thresh = 1e-6;

    FunctionDefaults<kDim>::set_k(k);
    FunctionDefaults<kDim>::set_thresh(thresh);
    FunctionDefaults<kDim>::set_refine(true);
    FunctionDefaults<kDim>::set_initial_level(2);
    FunctionDefaults<kDim>::set_truncate_mode(0);
    FunctionDefaults<kDim>::set_cubic_cell(-L / 2.0, L / 2.0);

    FunctionFactory<double, kDim> factory(world);
    Function<double, kDim> f = factory.f(gaussian);
    f.truncate();

    const std::string filename = "test_function_hdf5_v0_2d.h5";
    io::hdf5::save_function_v0(f, filename);
    world.gop.fence();

    Function<double, kDim> f2 = io::hdf5::load_function_v0<double>(world, filename);
    world.gop.fence();

    const double err = (f - f2).norm2();
    assert(err < 10.0 * thresh);

    world.gop.fence();
    finalize();
    return 0;
}