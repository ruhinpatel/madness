#include <FunctionIO.h>
#include <hdf5.h>
#include <fstream>
#include <iostream>
#include <madness/mra/mra.h>
#include <memory>
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

} // namespace

void test(World &world) {
  functionT fun = factoryT(world).f(f);
  fun.truncate();

  auto leafnodes = FunctionIO<double, D>::count_leaf_nodes(fun);
  if (world.rank() == 0) {
    print("fun: num leaf nodes: ", leafnodes);
  }

  {
    double norm = fun.norm2();
    if (world.rank() == 0) {
      std::cout << "norm = " << norm << std::endl;
      if (!write_norm_to_hdf5("fun.h5", norm)) {
        std::cerr << "failed to write HDF5 output file fun.h5" << std::endl;
      }
    }
    world.gop.fence();

    std::ofstream out("fun.dat", std::ios::out);
    fio::write_function(fun, out);
    out.close();
    // fun.print_tree();
  }

  {
    std::ifstream in("fun.dat", std::ios::in);
    functionT fun2 = fio::read_function(world, in);
    double norm = fun2.norm2();
    if (world.rank() == 0)
      std::cout << "norm = " << norm << std::endl;
    // write_function(fun2,std::cout);
    // fun2.print_tree();
    double err = (fun - fun2).norm2();
    if (world.rank() == 0)
      std::cout << "error = " << err << std::endl;
  }
}

int main(int argc, char **argv) {
  World &world = initialize(argc, argv);
  startup(world, argc, argv);
  std::cout.precision(6);

  FunctionDefaults<D>::set_k(k);
  FunctionDefaults<D>::set_thresh(thresh);
  FunctionDefaults<D>::set_refine(true);
  FunctionDefaults<D>::set_initial_level(2);
  FunctionDefaults<D>::set_truncate_mode(0);
  FunctionDefaults<D>::set_cubic_cell(-L / 2, L / 2);

  test(world);

  world.gop.fence();
  finalize();
  return 0;
}
