#include "funcdefaults.h"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <madness/mra/mra.h>
#include <string>
#include <vector>

#if defined(MADNESS_HAS_HDF5) && MADNESS_HAS_HDF5 && defined(__has_include)
#if __has_include(<hdf5.h>)
#include <hdf5.h>
#define MADNESS_WRITECOEFFS_HAS_HDF5_IO 1
#else
#define MADNESS_WRITECOEFFS_HAS_HDF5_IO 0
#endif
#elif defined(MADNESS_HAS_HDF5) && MADNESS_HAS_HDF5
#include <hdf5.h>
#define MADNESS_WRITECOEFFS_HAS_HDF5_IO 1
#else
#define MADNESS_WRITECOEFFS_HAS_HDF5_IO 0
#endif

using namespace madness;

template <typename T, std::size_t NDIM> struct FunctionIOData;

constexpr int simple_pow(int a, int b) {
  if (b == 0) {
    return 1;
  } else {
    int result = 1;
    for (int i = 0; i < b; i++) {
      result *= a;
    }
    return result;
  }
}

template <typename T, std::size_t NDIM> class FunctionIO {

private:
  long k = FunctionDefaults<NDIM>::get_k();
  long ndims = NDIM;
  long npts_per_box = simple_pow(k, ndims);

#if MADNESS_WRITECOEFFS_HAS_HDF5_IO
  template <typename ValueT>
  static bool hdf5_write_dataset(hid_t file_id, const char *name, hid_t type_id,
                                 const std::vector<hsize_t> &dims,
                                 const ValueT *data);

  template <typename ValueT>
  static bool hdf5_read_dataset(hid_t file_id, const char *name, hid_t type_id,
                                const std::vector<hsize_t> &expected_dims,
                                ValueT *data);
#endif

public:
  static size_t count_leaf_nodes(const Function<T, NDIM> &f) {
    const auto &coeffs = f.get_impl()->get_coeffs();
    size_t count = 0;
    for (auto it = coeffs.begin(); it != coeffs.end(); ++it) {
      // const auto &key = it->first;
      const auto &node = it->second;
      if (node.has_coeff()) {
        count++;
      }
    }
    f.get_impl()->world.gop.sum(count);
    return count;
  }
  static void write_function_coeffs(const Function<T, NDIM> &f,
                                    std::ostream &out, const Key<NDIM> &key) {
    const auto &coeffs = f.get_impl()->get_coeffs();
    auto it = coeffs.find(key).get();
    if (it == coeffs.end()) {
      for (int i = 0; i < key.level(); ++i)
        out << "  ";
      out << key << "  missing --> " << coeffs.owner(key) << "\n";
    } else {
      const auto &node = it->second;
      if (node.has_coeff()) {
        auto values = f.get_impl()->coeffs2values(key, node.coeff());
        for (int i = 0; i < key.level(); ++i)
          out << "  ";
        out << key.level() << " ";
        for (size_t i = 0; i < NDIM; ++i)
          out << key.translation()[i] << " ";
        out << std::endl;
#if HAVE_GENTENSOR
        MADNESS_EXCEPTION("FunctionIO not implemented for GenTensor", 0);
#else
        for (size_t i = 0; i < (size_t)values.size(); i++)
          out << values.ptr()[i] << " ";
#endif

        out << std::endl;
      }
      if (node.has_children()) {
        for (KeyChildIterator<NDIM> kit(key); kit; ++kit) {
          write_function_coeffs(f, out, kit.key());
        }
      }
    }
  }
  static void write_function(const Function<T, NDIM> &f, std::ostream &out) {
    f.reconstruct();
    std::cout << "NUMBER OF LEAF NODES: " << count_leaf_nodes(f) << std::endl;

    auto flags = out.flags();
    auto precision = out.precision();
    out << std::setprecision(17);
    out << std::scientific;

    if (f.get_impl()->world.rank() == 0) {
      out << NDIM << std::endl;
      const auto &cell = FunctionDefaults<NDIM>::get_cell();
      for (size_t d = 0; d < NDIM; ++d) {
        for (int i = 0; i < 2; ++i)
          out << cell(d, i) << " ";
        out << std::endl;
      }
      out << f.k() << std::endl;
      out << count_leaf_nodes(f) << std::endl;

      write_function_coeffs(f, out, Key<NDIM>(0));
    }
    f.get_impl()->world.gop.fence();

    out << std::setprecision(precision);
    out.setf(flags);
  }

  static void write_function_hdf5(const Function<T, NDIM> &f,
                                  const std::string &filename);

  static Function<T, NDIM> read_function_hdf5(World &world,
                                              const std::string &filename);

  static void read_function_coeffs(Function<T, NDIM> &f, std::istream &in,
                                   int num_leaf_nodes) {
    auto &coeffs = f.get_impl()->get_coeffs();

    for (int i = 0; i < num_leaf_nodes; i++) {
      Level n;
      Vector<Translation, NDIM> l;
      long dims[NDIM];
      in >> n;
      if (in.eof())
        break;

      for (size_t i = 0; i < NDIM; ++i) {
        in >> l[i];
        dims[i] = f.k();
      }
      Key<NDIM> key(n, l);

      Tensor<T> values(NDIM, dims);
      for (size_t i = 0; i < (size_t)values.size(); i++)
        in >> values.ptr()[i];
      auto t = f.get_impl()->values2coeffs(key, values);

      // f.get_impl()->accumulate2(t, coeffs, key);
      coeffs.task(key, &FunctionNode<T, NDIM>::accumulate2, t, coeffs, key);
    }
  }

  static Function<T, NDIM> read_function(World &world, std::istream &in) {
    size_t ndim;
    in >> ndim;
    MADNESS_CHECK(ndim == NDIM);

    Tensor<double> cell(NDIM, 2);
    for (size_t d = 0; d < NDIM; ++d) {
      for (int i = 0; i < 2; ++i)
        in >> cell(d, i);
    }
    FunctionDefaults<NDIM>::set_cell(cell);

    int k;
    in >> k;
    int num_leaf_nodes;
    in >> num_leaf_nodes;
    FunctionFactory<T, NDIM> factory(world);
    Function<T, NDIM> f(factory.k(k).empty());
    world.gop.fence();

    read_function_coeffs(f, in, num_leaf_nodes);

    f.verify_tree();

    return f;
  }
};
template <typename T, std::size_t NDIM> struct FunctionIOData {

  long k = 0;
  long npts_per_box = 0;
  std::size_t ndim = NDIM;
  std::array<std::pair<double, double>, NDIM> cell;
  long num_leaf_nodes{};
  std::vector<std::array<long, NDIM + 1>> nl;
  std::vector<std::vector<double>> values;

  FunctionIOData() = default;

  explicit FunctionIOData(const Function<T, NDIM> &f) {

    npts_per_box = simple_pow(f.k(), NDIM);

    f.reconstruct();
    if (f.get_impl()->world.rank() == 0) {
      num_leaf_nodes = FunctionIO<T, NDIM>::count_leaf_nodes(f);
      ndim = NDIM;
      k = f.k();
      const auto &cell_world = FunctionDefaults<NDIM>::get_cell();
      for (size_t d = 0; d < NDIM; ++d) {
        cell[d].first = cell_world(d, 0);
        cell[d].second = cell_world(d, 1);
      }

      initialize_func_coeffs(f, Key<NDIM>(0));
    }
    f.get_impl()->world.gop.fence();
  }

  void initialize_func_coeffs(const Function<T, NDIM> &f,
                              const Key<NDIM> &key) {
    const auto &coeffs = f.get_impl()->get_coeffs();
    auto it = coeffs.find(key).get();
    if (it == coeffs.end()) {
      for (int i = 0; i < key.level(); ++i)
        std::cout << "  ";
      std::cout << key << "  missing --> " << coeffs.owner(key) << "\n";
    } else {
      const auto &node = it->second;
      if (node.has_coeff()) {
        auto node_values = f.get_impl()->coeffs2values(key, node.coeff());
        std::array<long, NDIM + 1> key_i;
        key_i[0] = key.level();
        for (size_t i = 0; i < NDIM; ++i)
          key_i[i + 1] = key.translation()[i];
        nl.push_back(key_i);
        std::vector<double> values_i(npts_per_box);
#if HAVE_GENTENSOR
        MADNESS_EXCEPTION("FunctionIO coeffs not implemented for GenTensor", 0);
#else
        std::copy(node_values.ptr(), node_values.ptr() + npts_per_box,
                  values_i.begin());
#endif
        values.push_back(values_i);
      }
      if (node.has_children()) {
        for (KeyChildIterator<NDIM> kit(key); kit; ++kit) {
          initialize_func_coeffs(f, kit.key());
        }
      }
    }
  }
  void set_function_coeffs(Function<T, NDIM> &f, int num_leaf_nodes) {
    auto &coeffs = f.get_impl()->get_coeffs();

    for (int i = 0; i < num_leaf_nodes; i++) {
      Vector<Translation, NDIM> l;
      long dims[NDIM];

      for (size_t i = 0; i < NDIM; ++i) {
        dims[i] = f.k();
      }

      auto n = nl[i][0];
      for (size_t j = 0; j < NDIM; ++j) {
        l[j] = nl[i][j + 1];
      }
      Key<NDIM> key(n, l);

      Tensor<T> values(NDIM, dims);
      std::copy(this->values[i].begin(), this->values[i].end(), values.ptr());
      auto t = f.get_impl()->values2coeffs(key, values);

      // f.get_impl()->accumulate2(t, coeffs, key);
      coeffs.task(key, &FunctionNode<T, NDIM>::accumulate2, t, coeffs, key);
    }
  }

  Function<T, NDIM> create_function(World &world) {

    size_t ndim = this->ndim;
    MADNESS_CHECK(ndim == NDIM);
    Tensor<double> cell_t(NDIM, 2);
    for (size_t d = 0; d < NDIM; ++d) {
      cell_t(d, 0) = cell[d].first;
      cell_t(d, 1) = cell[d].second;
    }

    FunctionDefaults<NDIM>::set_cell(cell_t);

    FunctionFactory<T, NDIM> factory(world);
    Function<T, NDIM> f(factory.k(k).empty());
    world.gop.fence();

    set_function_coeffs(f, num_leaf_nodes);

    f.verify_tree();

    return f;
  }
};

template <typename T, std::size_t NDIM>
void to_json(json &j, const FunctionIOData<T, NDIM> &p) {
  j = json{{"npts_per_box", p.npts_per_box},
           {"k", p.k},
           {"cell", p.cell},
           {"num_leaf_nodes", p.num_leaf_nodes},
           {"nl", p.nl},
           {"ndim", p.ndim},
           {"values", p.values}};
}

template <typename T, std::size_t NDIM>
void from_json(const json &j, FunctionIOData<T, NDIM> &p) {
  j.at("npts_per_box").get_to(p.npts_per_box);
  j.at("k").get_to(p.k);
  j.at("cell").get_to(p.cell);
  j.at("num_leaf_nodes").get_to(p.num_leaf_nodes);
  j.at("nl").get_to(p.nl);
  j.at("values").get_to(p.values);
  j.at("ndim").get_to(p.ndim);
}

#if MADNESS_WRITECOEFFS_HAS_HDF5_IO
template <typename T, std::size_t NDIM>
template <typename ValueT>
bool FunctionIO<T, NDIM>::hdf5_write_dataset(
    hid_t file_id, const char *name, hid_t type_id,
    const std::vector<hsize_t> &dims, const ValueT *data) {
  hid_t dataspace_id =
      H5Screate_simple(static_cast<int>(dims.size()), dims.data(), nullptr);
  if (dataspace_id < 0) {
    return false;
  }

  hid_t dataset_id = H5Dcreate2(file_id, name, type_id, dataspace_id,
                                H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  if (dataset_id < 0) {
    H5Sclose(dataspace_id);
    return false;
  }

  const herr_t write_status =
      H5Dwrite(dataset_id, type_id, H5S_ALL, H5S_ALL, H5P_DEFAULT,
               static_cast<const void *>(data));
  const herr_t close_dataset_status = H5Dclose(dataset_id);
  const herr_t close_dataspace_status = H5Sclose(dataspace_id);

  return write_status >= 0 && close_dataset_status >= 0 &&
         close_dataspace_status >= 0;
}

template <typename T, std::size_t NDIM>
template <typename ValueT>
bool FunctionIO<T, NDIM>::hdf5_read_dataset(
    hid_t file_id, const char *name, hid_t type_id,
    const std::vector<hsize_t> &expected_dims, ValueT *data) {
  hid_t dataset_id = H5Dopen2(file_id, name, H5P_DEFAULT);
  if (dataset_id < 0) {
    return false;
  }

  hid_t dataspace_id = H5Dget_space(dataset_id);
  if (dataspace_id < 0) {
    H5Dclose(dataset_id);
    return false;
  }

  int rank = H5Sget_simple_extent_ndims(dataspace_id);
  if (rank < 0 || static_cast<std::size_t>(rank) != expected_dims.size()) {
    H5Sclose(dataspace_id);
    H5Dclose(dataset_id);
    return false;
  }

  std::vector<hsize_t> actual_dims(expected_dims.size(), 0);
  if (rank > 0) {
    const int dims_status =
        H5Sget_simple_extent_dims(dataspace_id, actual_dims.data(), nullptr);
    if (dims_status < 0) {
      H5Sclose(dataspace_id);
      H5Dclose(dataset_id);
      return false;
    }
    for (std::size_t i = 0; i < expected_dims.size(); ++i) {
      if (actual_dims[i] != expected_dims[i]) {
        H5Sclose(dataspace_id);
        H5Dclose(dataset_id);
        return false;
      }
    }
  }

  const herr_t read_status =
      H5Dread(dataset_id, type_id, H5S_ALL, H5S_ALL, H5P_DEFAULT,
              static_cast<void *>(data));
  const herr_t close_dataspace_status = H5Sclose(dataspace_id);
  const herr_t close_dataset_status = H5Dclose(dataset_id);

  return read_status >= 0 && close_dataset_status >= 0 &&
         close_dataspace_status >= 0;
}

template <typename T, std::size_t NDIM>
void FunctionIO<T, NDIM>::write_function_hdf5(const Function<T, NDIM> &f,
                                              const std::string &filename) {
  FunctionIOData<T, NDIM> data(f);
  auto &world = f.get_impl()->world;

  int fail = 0;
  if (world.rank() == 0) {
    if (data.num_leaf_nodes < 0 || data.npts_per_box <= 0 || data.k <= 0) {
      fail = 1;
    }

    const std::size_t leaf_count = static_cast<std::size_t>(data.num_leaf_nodes);
    const std::size_t npts_per_box =
        static_cast<std::size_t>(data.npts_per_box);
    if (data.nl.size() != leaf_count || data.values.size() != leaf_count) {
      fail = 1;
    }

    hid_t file_id = -1;
    if (fail == 0) {
      file_id =
          H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
      if (file_id < 0) {
        fail = 1;
      }
    }

    if (fail == 0) {
      const std::array<std::int64_t, 4> metadata = {
          static_cast<std::int64_t>(NDIM), static_cast<std::int64_t>(data.k),
          static_cast<std::int64_t>(npts_per_box),
          static_cast<std::int64_t>(leaf_count)};
      if (!hdf5_write_dataset(file_id, "meta", H5T_NATIVE_INT64,
                              {static_cast<hsize_t>(metadata.size())},
                              metadata.data())) {
        fail = 1;
      }
    }

    if (fail == 0) {
      std::array<double, NDIM * 2> cell_data{};
      for (std::size_t d = 0; d < NDIM; ++d) {
        cell_data[2 * d] = data.cell[d].first;
        cell_data[2 * d + 1] = data.cell[d].second;
      }
      if (!hdf5_write_dataset(
              file_id, "cell", H5T_NATIVE_DOUBLE,
              {static_cast<hsize_t>(NDIM), static_cast<hsize_t>(2)},
              cell_data.data())) {
        fail = 1;
      }
    }

    if (fail == 0 && leaf_count > 0) {
      std::vector<std::int64_t> key_data(leaf_count * (NDIM + 1), 0);
      std::vector<double> value_data(leaf_count * npts_per_box, 0.0);

      for (std::size_t i = 0; i < leaf_count; ++i) {
        if (data.values[i].size() != npts_per_box) {
          fail = 1;
          break;
        }
        for (std::size_t j = 0; j < NDIM + 1; ++j) {
          key_data[i * (NDIM + 1) + j] =
              static_cast<std::int64_t>(data.nl[i][j]);
        }
        std::copy(data.values[i].begin(), data.values[i].end(),
                  value_data.begin() + static_cast<std::ptrdiff_t>(i * npts_per_box));
      }

      if (fail == 0 &&
          !hdf5_write_dataset(
              file_id, "leaf_keys", H5T_NATIVE_INT64,
              {static_cast<hsize_t>(leaf_count), static_cast<hsize_t>(NDIM + 1)},
              key_data.data())) {
        fail = 1;
      }

      if (fail == 0 &&
          !hdf5_write_dataset(file_id, "values", H5T_NATIVE_DOUBLE,
                              {static_cast<hsize_t>(leaf_count),
                               static_cast<hsize_t>(npts_per_box)},
                              value_data.data())) {
        fail = 1;
      }
    }

    if (file_id >= 0 && H5Fclose(file_id) < 0) {
      fail = 1;
    }
  }

  world.gop.sum(fail);
  MADNESS_CHECK(fail == 0);
  world.gop.fence();
}

template <typename T, std::size_t NDIM>
Function<T, NDIM> FunctionIO<T, NDIM>::read_function_hdf5(
    World &world, const std::string &filename) {
  FunctionIOData<T, NDIM> data;
  int fail = 0;

  hid_t file_id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  if (file_id < 0) {
    fail = 1;
  }

  std::array<std::int64_t, 4> metadata{};
  if (fail == 0 &&
      !hdf5_read_dataset(file_id, "meta", H5T_NATIVE_INT64,
                         {static_cast<hsize_t>(metadata.size())},
                         metadata.data())) {
    fail = 1;
  }

  std::size_t leaf_count = 0;
  std::size_t npts_per_box = 0;
  if (fail == 0) {
    if (metadata[0] != static_cast<std::int64_t>(NDIM) || metadata[1] <= 0 ||
        metadata[2] <= 0 || metadata[3] < 0) {
      fail = 1;
    } else {
      leaf_count = static_cast<std::size_t>(metadata[3]);
      npts_per_box = static_cast<std::size_t>(metadata[2]);
    }
  }

  std::array<double, NDIM * 2> cell_data{};
  if (fail == 0 &&
      !hdf5_read_dataset(file_id, "cell", H5T_NATIVE_DOUBLE,
                         {static_cast<hsize_t>(NDIM), static_cast<hsize_t>(2)},
                         cell_data.data())) {
    fail = 1;
  }

  std::vector<std::int64_t> key_data;
  std::vector<double> value_data;
  if (fail == 0 && leaf_count > 0) {
    key_data.resize(leaf_count * (NDIM + 1));
    value_data.resize(leaf_count * npts_per_box);

    if (!hdf5_read_dataset(
            file_id, "leaf_keys", H5T_NATIVE_INT64,
            {static_cast<hsize_t>(leaf_count), static_cast<hsize_t>(NDIM + 1)},
            key_data.data())) {
      fail = 1;
    }

    if (fail == 0 &&
        !hdf5_read_dataset(file_id, "values", H5T_NATIVE_DOUBLE,
                           {static_cast<hsize_t>(leaf_count),
                            static_cast<hsize_t>(npts_per_box)},
                           value_data.data())) {
      fail = 1;
    }
  }

  if (file_id >= 0 && H5Fclose(file_id) < 0) {
    fail = 1;
  }

  world.gop.sum(fail);
  MADNESS_CHECK(fail == 0);

  data.ndim = NDIM;
  data.k = static_cast<long>(metadata[1]);
  data.npts_per_box = static_cast<long>(npts_per_box);
  data.num_leaf_nodes = static_cast<long>(leaf_count);
  data.nl.resize(leaf_count);
  data.values.resize(leaf_count);

  for (std::size_t d = 0; d < NDIM; ++d) {
    data.cell[d].first = cell_data[2 * d];
    data.cell[d].second = cell_data[2 * d + 1];
  }
  for (std::size_t i = 0; i < leaf_count; ++i) {
    for (std::size_t j = 0; j < NDIM + 1; ++j) {
      data.nl[i][j] = static_cast<long>(key_data[i * (NDIM + 1) + j]);
    }
    data.values[i].resize(npts_per_box);
    std::copy(
        value_data.begin() + static_cast<std::ptrdiff_t>(i * npts_per_box),
        value_data.begin() +
            static_cast<std::ptrdiff_t>((i + 1) * npts_per_box),
        data.values[i].begin());
  }

  return data.create_function(world);
}
#else
template <typename T, std::size_t NDIM>
void FunctionIO<T, NDIM>::write_function_hdf5(const Function<T, NDIM> &,
                                              const std::string &) {
  MADNESS_EXCEPTION("FunctionIO HDF5 support is unavailable in this build", 0);
}

template <typename T, std::size_t NDIM>
Function<T, NDIM> FunctionIO<T, NDIM>::read_function_hdf5(
    World &world, const std::string &) {
  MADNESS_EXCEPTION("FunctionIO HDF5 support is unavailable in this build", 0);
  FunctionFactory<T, NDIM> factory(world);
  return factory.empty();
}
#endif
