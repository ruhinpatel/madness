#include "function_hdf5_v0.h"

#include <hdf5.h>
#include <vector>
#include <array>
#include <string>
#include <cassert>
#include <type_traits>

namespace madness {
namespace io {
namespace hdf5 {

namespace {

constexpr int kDimension = 2;

void write_string_dataset(hid_t group, const char* name, const std::string& value) {
    hid_t space = H5Screate(H5S_SCALAR);
    assert(space >= 0);
    hid_t type = H5Tcopy(H5T_C_S1);
    assert(type >= 0);
    H5Tset_size(type, value.size());
    H5Tset_strpad(type, H5T_STR_NULLTERM);
    hid_t dset = H5Dcreate2(group, name, type, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    assert(dset >= 0);
    H5Dwrite(dset, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, value.c_str());
    H5Dclose(dset);
    H5Tclose(type);
    H5Sclose(space);
}

void write_scalar_int(hid_t group, const char* name, int value) {
    hid_t space = H5Screate(H5S_SCALAR);
    assert(space >= 0);
    hid_t dset = H5Dcreate2(group, name, H5T_NATIVE_INT, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    assert(dset >= 0);
    H5Dwrite(dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &value);
    H5Dclose(dset);
    H5Sclose(space);
}

void write_scalar_hsize(hid_t group, const char* name, hsize_t value) {
    hid_t space = H5Screate(H5S_SCALAR);
    assert(space >= 0);
    hid_t dset = H5Dcreate2(group, name, H5T_NATIVE_HSIZE, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    assert(dset >= 0);
    H5Dwrite(dset, H5T_NATIVE_HSIZE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &value);
    H5Dclose(dset);
    H5Sclose(space);
}

void write_cell_dataset(hid_t group, const char* name, const double cell[kDimension][kDimension]) {
    hsize_t dims[2] = {kDimension, kDimension};
    hid_t space = H5Screate_simple(2, dims, nullptr);
    assert(space >= 0);
    hid_t dset = H5Dcreate2(group, name, H5T_NATIVE_DOUBLE, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    assert(dset >= 0);
    H5Dwrite(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, cell);
    H5Dclose(dset);
    H5Sclose(space);
}

std::string read_string_dataset(hid_t group, const char* name) {
    hid_t dset = H5Dopen2(group, name, H5P_DEFAULT);
    assert(dset >= 0);
    hid_t type = H5Dget_type(dset);
    assert(type >= 0);
    size_t size = H5Tget_size(type);
    std::vector<char> buffer(size + 1, '\0');
    H5Dread(dset, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer.data());
    H5Tclose(type);
    H5Dclose(dset);
    return std::string(buffer.data());
}

int read_scalar_int(hid_t group, const char* name) {
    int value = 0;
    hid_t dset = H5Dopen2(group, name, H5P_DEFAULT);
    assert(dset >= 0);
    H5Dread(dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &value);
    H5Dclose(dset);
    return value;
}

hsize_t read_scalar_hsize(hid_t group, const char* name) {
    hsize_t value = 0;
    hid_t dset = H5Dopen2(group, name, H5P_DEFAULT);
    assert(dset >= 0);
    H5Dread(dset, H5T_NATIVE_HSIZE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &value);
    H5Dclose(dset);
    return value;
}

void read_cell_dataset(hid_t group, const char* name, double cell[kDimension][kDimension]) {
    hid_t dset = H5Dopen2(group, name, H5P_DEFAULT);
    assert(dset >= 0);
    H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, cell);
    H5Dclose(dset);
}

}  // namespace

template <typename T>
void save_function_v0(
    const Function<T, 2>& f,
    const std::string& filename
) {
    static_assert(std::is_floating_point<T>::value, "save_function_v0 supports real-valued functions only.");

    f.reconstruct();

    std::vector<int> leaf_levels;
    std::vector<std::array<int, 2>> leaf_translations;
    std::vector<double> leaf_values;

    if (f.get_impl()->world.rank() == 0) {
        const auto& coeffs = f.get_impl()->get_coeffs();
        std::vector<Key<2>> stack;
        stack.push_back(Key<2>(0));

        while (!stack.empty()) {
            Key<2> key = stack.back();
            stack.pop_back();

            auto it = coeffs.find(key).get();
            if (it == coeffs.end()) continue;

            const auto& node = it->second;
            if (node.has_coeff()) {
                leaf_levels.push_back(key.level());
                leaf_translations.push_back({{key.translation()[0], key.translation()[1]}});

                auto values = f.get_impl()->coeffs2values(key, node.coeff());
                for (size_t i = 0; i < values.size(); ++i) {
                    leaf_values.push_back(static_cast<double>(values.ptr()[i]));
                }
            }

            if (node.has_children()) {
                for (KeyChildIterator<2> kit(key); kit; ++kit) {
                    stack.push_back(kit.key());
                }
            }
        }

        hid_t file = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
        assert(file >= 0);
        hid_t metadata = H5Gcreate2(file, "/metadata", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        assert(metadata >= 0);
        hid_t function = H5Gcreate2(file, "/function", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        assert(function >= 0);

        write_string_dataset(metadata, "schema_version", "v0");
        write_scalar_int(metadata, "dimension", kDimension);
        write_string_dataset(metadata, "value_type", "real");
        write_scalar_int(metadata, "k", static_cast<int>(f.k()));

        const auto& cell = FunctionDefaults<2>::get_cell();
        double cell_data[kDimension][kDimension] = {};
        for (int d = 0; d < kDimension; ++d) {
            for (int i = 0; i < kDimension; ++i) {
                cell_data[d][i] = cell(d, i);
            }
        }
        write_cell_dataset(metadata, "cell", cell_data);

        hsize_t num_leaf_nodes = static_cast<hsize_t>(leaf_levels.size());
        write_scalar_hsize(metadata, "num_leaf_nodes", num_leaf_nodes);

        const hsize_t n = num_leaf_nodes;
        if (n > 0) {
            hsize_t level_dims[1] = {n};
            hid_t level_space = H5Screate_simple(1, level_dims, nullptr);
            hid_t level_dset = H5Dcreate2(function, "leaf_level", H5T_NATIVE_INT, level_space,
                                          H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            assert(level_dset >= 0);
            H5Dwrite(level_dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, leaf_levels.data());
            H5Dclose(level_dset);
            H5Sclose(level_space);

            std::vector<int> translations_flat;
            translations_flat.reserve(leaf_translations.size() * kDimension);
            for (const auto& t : leaf_translations) {
                translations_flat.push_back(t[0]);
                translations_flat.push_back(t[1]);
            }
            hsize_t translation_dims[2] = {n, kDimension};
            hid_t translation_space = H5Screate_simple(2, translation_dims, nullptr);
            hid_t translation_dset = H5Dcreate2(function, "leaf_translation", H5T_NATIVE_INT,
                                                translation_space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            assert(translation_dset >= 0);
            H5Dwrite(translation_dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                     translations_flat.data());
            H5Dclose(translation_dset);
            H5Sclose(translation_space);

            const hsize_t kk = static_cast<hsize_t>(f.k() * f.k());
            hsize_t values_dims[2] = {n, kk};
            hid_t values_space = H5Screate_simple(2, values_dims, nullptr);
            hid_t values_dset = H5Dcreate2(function, "leaf_values", H5T_NATIVE_DOUBLE,
                                           values_space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            assert(values_dset >= 0);
            H5Dwrite(values_dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, leaf_values.data());
            H5Dclose(values_dset);
            H5Sclose(values_space);
        } else {
            hsize_t level_dims[1] = {0};
            hid_t level_space = H5Screate_simple(1, level_dims, nullptr);
            hid_t level_dset = H5Dcreate2(function, "leaf_level", H5T_NATIVE_INT, level_space,
                                          H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            H5Dclose(level_dset);
            H5Sclose(level_space);

            hsize_t translation_dims[2] = {0, kDimension};
            hid_t translation_space = H5Screate_simple(2, translation_dims, nullptr);
            hid_t translation_dset = H5Dcreate2(function, "leaf_translation", H5T_NATIVE_INT,
                                                translation_space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            H5Dclose(translation_dset);
            H5Sclose(translation_space);

            const hsize_t kk = static_cast<hsize_t>(f.k() * f.k());
            hsize_t values_dims[2] = {0, kk};
            hid_t values_space = H5Screate_simple(2, values_dims, nullptr);
            hid_t values_dset = H5Dcreate2(function, "leaf_values", H5T_NATIVE_DOUBLE,
                                           values_space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            H5Dclose(values_dset);
            H5Sclose(values_space);
        }

        H5Gclose(function);
        H5Gclose(metadata);
        H5Fclose(file);
    }

    f.get_impl()->world.gop.fence();
}

template <typename T>
Function<T, 2> load_function_v0(
    World& world,
    const std::string& filename
) {
    static_assert(std::is_floating_point<T>::value, "load_function_v0 supports real-valued functions only.");

    int k = 0;
    hsize_t num_leaf_nodes = 0;
    double cell_data[kDimension][kDimension] = {};
    std::vector<int> leaf_levels;
    std::vector<int> leaf_translations_flat;
    std::vector<double> leaf_values;

    if (world.rank() == 0) {
        hid_t file = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        assert(file >= 0);
        hid_t metadata = H5Gopen2(file, "/metadata", H5P_DEFAULT);
        assert(metadata >= 0);
        hid_t function = H5Gopen2(file, "/function", H5P_DEFAULT);
        assert(function >= 0);

        const std::string schema_version = read_string_dataset(metadata, "schema_version");
        assert(schema_version == "v0");
        const std::string value_type = read_string_dataset(metadata, "value_type");
        assert(value_type == "real");
        const int dimension = read_scalar_int(metadata, "dimension");
        assert(dimension == kDimension);

        k = read_scalar_int(metadata, "k");
        read_cell_dataset(metadata, "cell", cell_data);
        num_leaf_nodes = read_scalar_hsize(metadata, "num_leaf_nodes");

        const hsize_t n = num_leaf_nodes;
        if (n > 0) {
            leaf_levels.resize(static_cast<size_t>(n));
            leaf_translations_flat.resize(static_cast<size_t>(n) * kDimension);
            leaf_values.resize(static_cast<size_t>(n) * static_cast<size_t>(k) * static_cast<size_t>(k));

            hid_t level_dset = H5Dopen2(function, "leaf_level", H5P_DEFAULT);
            assert(level_dset >= 0);
            H5Dread(level_dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, leaf_levels.data());
            H5Dclose(level_dset);

            hid_t translation_dset = H5Dopen2(function, "leaf_translation", H5P_DEFAULT);
            assert(translation_dset >= 0);
            H5Dread(translation_dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                    leaf_translations_flat.data());
            H5Dclose(translation_dset);

            hid_t values_dset = H5Dopen2(function, "leaf_values", H5P_DEFAULT);
            assert(values_dset >= 0);
            H5Dread(values_dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, leaf_values.data());
            H5Dclose(values_dset);
        }

        H5Gclose(function);
        H5Gclose(metadata);
        H5Fclose(file);
    }

    world.gop.broadcast(k, 0);
    world.gop.broadcast(num_leaf_nodes, 0);
    world.gop.broadcast(&cell_data[0][0], kDimension * kDimension, 0);

    if (world.rank() != 0) {
        leaf_levels.resize(static_cast<size_t>(num_leaf_nodes));
        leaf_translations_flat.resize(static_cast<size_t>(num_leaf_nodes) * kDimension);
        leaf_values.resize(static_cast<size_t>(num_leaf_nodes) * static_cast<size_t>(k) * static_cast<size_t>(k));
    }

    if (num_leaf_nodes > 0) {
        world.gop.broadcast(leaf_levels.data(), leaf_levels.size(), 0);
        world.gop.broadcast(leaf_translations_flat.data(), leaf_translations_flat.size(), 0);
        world.gop.broadcast(leaf_values.data(), leaf_values.size(), 0);
    }

    Tensor<double> cell_tensor(kDimension, 2);
    for (int d = 0; d < kDimension; ++d) {
        for (int i = 0; i < kDimension; ++i) {
            cell_tensor(d, i) = cell_data[d][i];
        }
    }
    FunctionDefaults<2>::set_cell(cell_tensor);

    FunctionFactory<T, 2> factory(world);
    Function<T, 2> f = factory.k(k).empty();
    world.gop.fence();

    auto& coeffs = f.get_impl()->get_coeffs();
    const size_t kk = static_cast<size_t>(k) * static_cast<size_t>(k);
    for (size_t idx = 0; idx < static_cast<size_t>(num_leaf_nodes); ++idx) {
        Level level = static_cast<Level>(leaf_levels[idx]);
        Vector<Translation, 2> translation;
        translation[0] = static_cast<Translation>(leaf_translations_flat[idx * kDimension]);
        translation[1] = static_cast<Translation>(leaf_translations_flat[idx * kDimension + 1]);
        Key<2> key(level, translation);

        long dims[kDimension] = {k, k};
        Tensor<T> values(kDimension, dims);
        for (size_t i = 0; i < kk; ++i) {
            values.ptr()[i] = static_cast<T>(leaf_values[idx * kk + i]);
        }
        auto t = f.get_impl()->values2coeffs(key, values);
        coeffs.task(key, &FunctionNode<T, 2>::accumulate2, t, coeffs, key);
    }

    world.gop.fence();
    f.verify_tree();

    return f;
}

template void save_function_v0<double>(const Function<double, 2>& f, const std::string& filename);
template Function<double, 2> load_function_v0<double>(World& world, const std::string& filename);

}  // namespace hdf5
}  // namespace io
}  // namespace madness
