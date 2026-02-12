#include <FunctionIO.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <madness/mra/mra.h>
#include <memory>
#include <sstream>
#include <string>

using namespace madness;

static const size_t D = 2;
typedef Vector<double, D> coordT;
typedef Key<D> keyT;
typedef double dataT; // was std::complex<double>
typedef std::shared_ptr<FunctionFunctorInterface<dataT, D>> functorT;
typedef Function<dataT, D> functionT;
typedef FunctionFactory<dataT, D> factoryT;
typedef SeparatedConvolution<dataT, D> operatorT;

static const double L = 4.0;
static const long k = 5;           // wavelet order
static const double thresh = 1e-3; // precision

static dataT f(const coordT &r) {
  double R = r.normf();
  return std::exp(-R * R);
}

using fio = FunctionIO<double, D>;

namespace {

struct RunOptions {
  std::string h5_filename = "fun.h5";
  std::string dat_filename = "fun.dat";
  double max_roundtrip_error = 1.0e-8;
};

bool parse_options(int argc, char **argv, RunOptions &options) {
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    if (arg.rfind("--h5=", 0) == 0) {
      options.h5_filename = arg.substr(5);
    } else if (arg.rfind("--dat=", 0) == 0) {
      options.dat_filename = arg.substr(6);
    } else if (arg.rfind("--max-roundtrip-error=", 0) == 0) {
      const std::string value = arg.substr(22);
      try {
        options.max_roundtrip_error = std::stod(value);
      } catch (const std::exception &) {
        std::cerr << "invalid numeric value for --max-roundtrip-error: " << value
                  << std::endl;
        return false;
      }
      if (!(options.max_roundtrip_error > 0.0)) {
        std::cerr << "--max-roundtrip-error must be > 0" << std::endl;
        return false;
      }
    } else {
      std::cerr << "unknown argument: " << arg << std::endl;
      return false;
    }
  }
  return true;
}

} // namespace

bool test(World &world, const RunOptions &options) {
  functionT fun = factoryT(world).f(f);
  fun.truncate();

  const auto leafnodes = FunctionIO<double, D>::count_leaf_nodes(fun);
  if (world.rank() == 0) {
    print("fun: num leaf nodes:", leafnodes);
  }

  int fail = 0;
  const double norm = fun.norm2();

  {
    if (world.rank() == 0)
      std::cout << "norm = " << norm << std::endl;

    fio::write_function_hdf5(fun, options.h5_filename);

    std::ofstream out;
    std::ostringstream sink;
    std::ostream *output = &sink;
    if (world.rank() == 0) {
      out.open(options.dat_filename, std::ios::out | std::ios::trunc);
      if (!out) {
        std::cerr << "failed to open text output file " << options.dat_filename
                  << std::endl;
        fail = 1;
      } else {
        output = &out;
      }
    }
    world.gop.sum(fail);
    if (fail != 0) {
      return false;
    }

    fio::write_function(fun, *output);
    if (world.rank() == 0) {
      out.close();
    }
  }

  world.gop.fence();

  {
    std::ifstream in(options.dat_filename, std::ios::in);
    if (!in) {
      std::cerr << "rank " << world.rank() << " failed to open "
                << options.dat_filename << " for readback" << std::endl;
      fail = 1;
    }
    world.gop.sum(fail);
    if (fail != 0) {
      return false;
    }

    functionT fun2 = fio::read_function(world, in);
    in.close();

    functionT fun_hdf5 = fio::read_function_hdf5(world, options.h5_filename);

    const double text_readback_norm = fun2.norm2();
    const double hdf5_readback_norm = fun_hdf5.norm2();
    const double text_roundtrip_error = (fun - fun2).norm2();
    const double hdf5_roundtrip_error = (fun - fun_hdf5).norm2();

    if (world.rank() == 0) {
      std::cout << "norm(text readback) = " << text_readback_norm << std::endl;
      std::cout << "norm(hdf5 readback) = " << hdf5_readback_norm << std::endl;
      std::cout << "text roundtrip error = " << text_roundtrip_error
                << std::endl;
      std::cout << "hdf5 roundtrip error = " << hdf5_roundtrip_error
                << std::endl;

      if (!std::isfinite(text_roundtrip_error) ||
          text_roundtrip_error > options.max_roundtrip_error) {
        std::cerr << "text roundtrip error exceeds tolerance ("
                  << options.max_roundtrip_error << ")" << std::endl;
        fail = 1;
      }

      if (!std::isfinite(hdf5_roundtrip_error) ||
          hdf5_roundtrip_error > options.max_roundtrip_error) {
        std::cerr << "hdf5 roundtrip error exceeds tolerance ("
                  << options.max_roundtrip_error << ")" << std::endl;
        fail = 1;
      }
    }
    world.gop.sum(fail);
    if (fail != 0) {
      return false;
    }
  }

  return true;
}

int main(int argc, char **argv) {
  RunOptions options;
  if (!parse_options(argc, argv, options)) {
    std::cerr << "usage: writecoeff_hdf5 [--h5=<file.h5>] [--dat=<file.dat>] "
                 "[--max-roundtrip-error=<positive float>]"
              << std::endl;
    return 2;
  }

  World &world = initialize(argc, argv);
  startup(world, argc, argv);
  std::cout.precision(6);

  FunctionDefaults<D>::set_k(k);
  FunctionDefaults<D>::set_thresh(thresh);
  FunctionDefaults<D>::set_refine(true);
  FunctionDefaults<D>::set_initial_level(2);
  FunctionDefaults<D>::set_truncate_mode(0);
  FunctionDefaults<D>::set_cubic_cell(-L / 2, L / 2);

  const bool ok = test(world, options);

  world.gop.fence();
  finalize();
  return ok ? 0 : 1;
}
