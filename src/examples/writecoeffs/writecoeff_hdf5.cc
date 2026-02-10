#include <FunctionIO.h>
#include <hdf5.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <madness/mra/mra.h>
#include <memory>
#include <sstream>
#include <string>
#include <limits>

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

bool write_norm_to_hdf5(const std::string &filename, double norm) {
  const hsize_t dims[1] = {1};

  hid_t file =
      H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  if (file < 0) {
    return false;
  }

  hid_t dataspace = H5Screate_simple(1, dims, nullptr);
  if (dataspace < 0) {
    H5Fclose(file);
    return false;
  }

  hid_t dataset =
      H5Dcreate2(file, "norm2", H5T_NATIVE_DOUBLE, dataspace, H5P_DEFAULT,
                 H5P_DEFAULT, H5P_DEFAULT);
  if (dataset < 0) {
    H5Sclose(dataspace);
    H5Fclose(file);
    return false;
  }

  const herr_t write_status =
      H5Dwrite(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &norm);
  const herr_t close_dataset_status = H5Dclose(dataset);
  const herr_t close_dataspace_status = H5Sclose(dataspace);
  const herr_t close_file_status = H5Fclose(file);

  return write_status >= 0 && close_dataset_status >= 0 &&
         close_dataspace_status >= 0 && close_file_status >= 0;
}

bool read_norm_from_hdf5(const std::string &filename, double &norm) {
  hid_t file = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  if (file < 0) {
    return false;
  }

  hid_t dataset = H5Dopen2(file, "norm2", H5P_DEFAULT);
  if (dataset < 0) {
    H5Fclose(file);
    return false;
  }

  const herr_t read_status =
      H5Dread(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &norm);
  const herr_t close_dataset_status = H5Dclose(dataset);
  const herr_t close_file_status = H5Fclose(file);

  return read_status >= 0 && close_dataset_status >= 0 && close_file_status >= 0;
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
    if (world.rank() == 0) {
      std::cout << "norm = " << norm << std::endl;
      if (!write_norm_to_hdf5(options.h5_filename, norm)) {
        std::cerr << "failed to write HDF5 output file " << options.h5_filename
                  << std::endl;
        fail = 1;
      }
    }
    world.gop.sum(fail);
    if (fail != 0) {
      return false;
    }

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

    const double readback_norm = fun2.norm2();
    const double roundtrip_error = (fun - fun2).norm2();

    if (world.rank() == 0) {
      std::cout << "norm(readback) = " << readback_norm << std::endl;
      std::cout << "roundtrip error = " << roundtrip_error << std::endl;
      if (!std::isfinite(roundtrip_error) ||
          roundtrip_error > options.max_roundtrip_error) {
        std::cerr << "roundtrip error exceeds tolerance ("
                  << options.max_roundtrip_error << ")" << std::endl;
        fail = 1;
      }

      double h5_norm = 0.0;
      if (!read_norm_from_hdf5(options.h5_filename, h5_norm)) {
        std::cerr << "failed to read back HDF5 file " << options.h5_filename
                  << std::endl;
        fail = 1;
      } else {
        const double h5_diff = std::abs(h5_norm - norm);
        const double h5_tol = 32.0 * std::numeric_limits<double>::epsilon() *
                              std::max(1.0, std::abs(norm));
        std::cout << "hdf5 norm difference = " << h5_diff << std::endl;
        if (h5_diff > h5_tol) {
          std::cerr << "HDF5 readback mismatch exceeds tolerance (" << h5_tol
                    << ")" << std::endl;
          fail = 1;
        }
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
