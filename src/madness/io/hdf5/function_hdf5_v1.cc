#include "function_hdf5_v1.h"

#include <hdf5.h>
#include <vector>
#include <string>
#include <cassert>
#include <cstddef>
#include <type_traits>

namespace madness {
namespace io {
namespace hdf5 {

// ============================================================
// Internal helpers
// ============================================================
namespace {

// --- scalar writers ---

void write_string_ds(hid_t loc, const char* name, const std::string& v) {
    hid_t space = H5Screate(H5S_SCALAR);
    hid_t type  = H5Tcopy(H5T_C_S1);
    H5Tset_size(type, v.size());
    H5Tset_strpad(type, H5T_STR_NULLTERM);
    hid_t dset = H5Dcreate2(loc, name, type, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    assert(dset >= 0);
    H5Dwrite(dset, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, v.c_str());
    H5Dclose(dset); H5Tclose(type); H5Sclose(space);
}

void write_int_scalar(hid_t loc, const char* name, int v) {
    hid_t space = H5Screate(H5S_SCALAR);
    hid_t dset  = H5Dcreate2(loc, name, H5T_NATIVE_INT, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    assert(dset >= 0);
    H5Dwrite(dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &v);
    H5Dclose(dset); H5Sclose(space);
}

void write_hsize_scalar(hid_t loc, const char* name, hsize_t v) {
    hid_t space = H5Screate(H5S_SCALAR);
    hid_t dset  = H5Dcreate2(loc, name, H5T_NATIVE_HSIZE, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    assert(dset >= 0);
    H5Dwrite(dset, H5T_NATIVE_HSIZE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &v);
    H5Dclose(dset); H5Sclose(space);
}

// --- scalar readers ---

std::string read_string_ds(hid_t loc, const char* name) {
    hid_t dset = H5Dopen2(loc, name, H5P_DEFAULT);
    assert(dset >= 0);
    hid_t type = H5Dget_type(dset);
    size_t sz  = H5Tget_size(type);
    std::vector<char> buf(sz + 1, '\0');
    H5Dread(dset, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf.data());
    H5Tclose(type); H5Dclose(dset);
    return std::string(buf.data());
}

int read_int_scalar(hid_t loc, const char* name) {
    int v = 0;
    hid_t dset = H5Dopen2(loc, name, H5P_DEFAULT);
    assert(dset >= 0);
    H5Dread(dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &v);
    H5Dclose(dset);
    return v;
}

hsize_t read_hsize_scalar(hid_t loc, const char* name) {
    hsize_t v = 0;
    hid_t dset = H5Dopen2(loc, name, H5P_DEFAULT);
    assert(dset >= 0);
    H5Dread(dset, H5T_NATIVE_HSIZE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &v);
    H5Dclose(dset);
    return v;
}

// --- cell: stored as [NDIM][2]  (lo, hi per dimension) ---
//
// v0 bug: stored as [NDIM][NDIM], which worked by accident only when NDIM=2.
// Here we fix this: the second axis is always 2 (lo and hi).

void write_cell(hid_t loc, const char* name,
                const std::vector<double>& flat, int ndim) {
    // flat layout: lo_0, hi_0, lo_1, hi_1, ..., lo_{n-1}, hi_{n-1}
    hsize_t dims[2] = { static_cast<hsize_t>(ndim), 2 };
    hid_t space = H5Screate_simple(2, dims, nullptr);
    hid_t dset  = H5Dcreate2(loc, name, H5T_NATIVE_DOUBLE, space,
                              H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    assert(dset >= 0);
    H5Dwrite(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, flat.data());
    H5Dclose(dset); H5Sclose(space);
}

void read_cell(hid_t loc, const char* name,
               std::vector<double>& flat, int ndim) {
    flat.resize(static_cast<size_t>(ndim) * 2);
    hid_t dset = H5Dopen2(loc, name, H5P_DEFAULT);
    assert(dset >= 0);
    H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, flat.data());
    H5Dclose(dset);
}

// --- 1-D int dataset ---

void write_int1d(hid_t loc, const char* name, const std::vector<int>& v) {
    hsize_t n = static_cast<hsize_t>(v.size());
    hsize_t dims[1] = { n };
    hid_t space = H5Screate_simple(1, dims, nullptr);
    hid_t dset  = H5Dcreate2(loc, name, H5T_NATIVE_INT, space,
                              H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    assert(dset >= 0);
    if (n > 0)
        H5Dwrite(dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, v.data());
    H5Dclose(dset); H5Sclose(space);
}

void read_int1d(hid_t loc, const char* name, std::vector<int>& v) {
    hid_t dset  = H5Dopen2(loc, name, H5P_DEFAULT);
    assert(dset >= 0);
    hid_t space = H5Dget_space(dset);
    hsize_t n;
    H5Sget_simple_extent_dims(space, &n, nullptr);
    v.resize(static_cast<size_t>(n));
    if (n > 0)
        H5Dread(dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, v.data());
    H5Sclose(space); H5Dclose(dset);
}

// --- 2-D int dataset: leaf_translation [N][NDIM] ---

void write_int2d(hid_t loc, const char* name, const std::vector<int>& flat,
                 hsize_t rows, hsize_t cols) {
    hsize_t dims[2] = { rows, cols };
    hid_t space = H5Screate_simple(2, dims, nullptr);
    hid_t dset  = H5Dcreate2(loc, name, H5T_NATIVE_INT, space,
                              H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    assert(dset >= 0);
    if (rows > 0 && cols > 0)
        H5Dwrite(dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, flat.data());
    H5Dclose(dset); H5Sclose(space);
}

void read_int2d(hid_t loc, const char* name, std::vector<int>& flat,
                hsize_t& rows, hsize_t& cols) {
    hid_t dset  = H5Dopen2(loc, name, H5P_DEFAULT);
    assert(dset >= 0);
    hid_t space = H5Dget_space(dset);
    hsize_t dims[2];
    H5Sget_simple_extent_dims(space, dims, nullptr);
    rows = dims[0]; cols = dims[1];
    flat.resize(static_cast<size_t>(rows) * static_cast<size_t>(cols));
    if (rows > 0 && cols > 0)
        H5Dread(dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, flat.data());
    H5Sclose(space); H5Dclose(dset);
}

// --- 2-D double dataset: leaf_values [N][k^NDIM] ---

void write_dbl2d(hid_t loc, const char* name, const std::vector<double>& flat,
                 hsize_t rows, hsize_t cols) {
    hsize_t dims[2] = { rows, cols };
    hid_t space = H5Screate_simple(2, dims, nullptr);
    hid_t dset  = H5Dcreate2(loc, name, H5T_NATIVE_DOUBLE, space,
                              H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    assert(dset >= 0);
    if (rows > 0 && cols > 0)
        H5Dwrite(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, flat.data());
    H5Dclose(dset); H5Sclose(space);
}

void read_dbl2d(hid_t loc, const char* name, std::vector<double>& flat,
                hsize_t& rows, hsize_t& cols) {
    hid_t dset  = H5Dopen2(loc, name, H5P_DEFAULT);
    assert(dset >= 0);
    hid_t space = H5Dget_space(dset);
    hsize_t dims[2];
    H5Sget_simple_extent_dims(space, dims, nullptr);
    rows = dims[0]; cols = dims[1];
    flat.resize(static_cast<size_t>(rows) * static_cast<size_t>(cols));
    if (rows > 0 && cols > 0)
        H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, flat.data());
    H5Sclose(space); H5Dclose(dset);
}

// --- k^NDIM: runtime product (k is known at runtime, NDIM at compile time) ---

inline size_t kpow(int k, std::size_t ndim) {
    size_t r = 1;
    for (std::size_t d = 0; d < ndim; ++d) r *= static_cast<size_t>(k);
    return r;
}

}  // anonymous namespace


// ============================================================
// save_function_v1
// ============================================================

template <typename T, std::size_t NDIM>
void save_function_v1(const Function<T, NDIM>& f, const std::string& filename) {

    static_assert(std::is_floating_point<T>::value,
                  "save_function_v1: only real-valued (float/double) supported.");

    f.reconstruct();

    std::vector<int>    leaf_levels;
    std::vector<int>    leaf_trans_flat;   // [N * NDIM]
    std::vector<double> leaf_vals_flat;    // [N * k^NDIM]

    if (f.get_impl()->world.rank() == 0) {

        const int    k   = static_cast<int>(f.k());
        const size_t vpn = kpow(k, NDIM);  // values per node

        // DFS over the reconstructed tree from the root
        std::vector<Key<NDIM>> stack;
        stack.push_back(Key<NDIM>(0));

        const auto& coeffs = f.get_impl()->get_coeffs();

        while (!stack.empty()) {
            Key<NDIM> key = stack.back();
            stack.pop_back();

            auto it = coeffs.find(key).get();
            if (it == coeffs.end()) continue;

            const auto& node = it->second;

            if (node.has_coeff()) {
                // level
                leaf_levels.push_back(static_cast<int>(key.level()));

                // translation: one int per spatial dimension
                const auto& tr = key.translation();
                for (std::size_t d = 0; d < NDIM; ++d)
                    leaf_trans_flat.push_back(static_cast<int>(tr[d]));

                // scaling-function coefficients -> real-space values on the
                // k^NDIM Gauss-Legendre quadrature grid
                auto vals = f.get_impl()->coeffs2values(key, node.coeff());
                const double* ptr = vals.ptr();
                for (size_t i = 0; i < vpn; ++i)
                    leaf_vals_flat.push_back(static_cast<double>(ptr[i]));
            }

            if (node.has_children()) {
                for (KeyChildIterator<NDIM> kit(key); kit; ++kit)
                    stack.push_back(kit.key());
            }
        }

        // simulation box: FunctionDefaults<NDIM>::get_cell() returns
        // Tensor<double> of shape [NDIM][2]  (lo, hi per dimension)
        const auto& cell = FunctionDefaults<NDIM>::get_cell();
        std::vector<double> cell_flat(NDIM * 2);
        for (std::size_t d = 0; d < NDIM; ++d) {
            cell_flat[d * 2    ] = cell(static_cast<long>(d), 0L);  // lo
            cell_flat[d * 2 + 1] = cell(static_cast<long>(d), 1L);  // hi
        }

        const hsize_t N   = static_cast<hsize_t>(leaf_levels.size());
        const hsize_t VPN = static_cast<hsize_t>(vpn);

        hid_t file     = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC,
                                   H5P_DEFAULT, H5P_DEFAULT);
        hid_t meta_grp = H5Gcreate2(file, "/metadata",
                                    H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        hid_t func_grp = H5Gcreate2(file, "/function",
                                    H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        assert(file >= 0 && meta_grp >= 0 && func_grp >= 0);

        write_string_ds   (meta_grp, "schema_version", "v1");
        write_int_scalar  (meta_grp, "dimension",      static_cast<int>(NDIM));
        write_string_ds   (meta_grp, "value_type",     "real");
        write_int_scalar  (meta_grp, "k",              k);
        write_cell        (meta_grp, "cell",           cell_flat, static_cast<int>(NDIM));
        write_hsize_scalar(meta_grp, "num_leaf_nodes", N);

        write_int1d (func_grp, "leaf_level",       leaf_levels);
        write_int2d (func_grp, "leaf_translation", leaf_trans_flat, N, NDIM);
        write_dbl2d (func_grp, "leaf_values",      leaf_vals_flat,  N, VPN);

        H5Gclose(func_grp);
        H5Gclose(meta_grp);
        H5Fclose(file);
    }

    f.get_impl()->world.gop.fence();
}


// ============================================================
// load_function_v1
// ============================================================

template <typename T, std::size_t NDIM>
Function<T, NDIM> load_function_v1(World& world, const std::string& filename) {

    static_assert(std::is_floating_point<T>::value,
                  "load_function_v1: only real-valued (float/double) supported.");

    int     k              = 0;
    hsize_t num_leaf_nodes = 0;
    std::vector<double> cell_flat;
    std::vector<int>    leaf_levels;
    std::vector<int>    leaf_trans_flat;
    std::vector<double> leaf_vals_flat;

    if (world.rank() == 0) {
        hid_t file     = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        hid_t meta_grp = H5Gopen2(file, "/metadata", H5P_DEFAULT);
        hid_t func_grp = H5Gopen2(file, "/function",  H5P_DEFAULT);
        assert(file >= 0 && meta_grp >= 0 && func_grp >= 0);

        // validate schema
        assert(read_string_ds(meta_grp, "schema_version") == "v1");
        assert(read_string_ds(meta_grp, "value_type")     == "real");
        assert(static_cast<std::size_t>(
                   read_int_scalar(meta_grp, "dimension")) == NDIM);

        k              = read_int_scalar  (meta_grp, "k");
        num_leaf_nodes = read_hsize_scalar(meta_grp, "num_leaf_nodes");
        read_cell(meta_grp, "cell", cell_flat, static_cast<int>(NDIM));

        read_int1d(func_grp, "leaf_level", leaf_levels);

        hsize_t tr_rows, tr_cols;
        read_int2d(func_grp, "leaf_translation", leaf_trans_flat, tr_rows, tr_cols);
        assert(tr_rows == num_leaf_nodes && tr_cols == NDIM);

        hsize_t val_rows, val_cols;
        read_dbl2d(func_grp, "leaf_values", leaf_vals_flat, val_rows, val_cols);
        assert(val_rows == num_leaf_nodes && val_cols == kpow(k, NDIM));

        H5Gclose(func_grp);
        H5Gclose(meta_grp);
        H5Fclose(file);
    }

    // ---- broadcast to all MPI ranks ----

    world.gop.broadcast(k,              0);
    world.gop.broadcast(num_leaf_nodes, 0);

    if (world.rank() != 0) cell_flat.resize(NDIM * 2);
    world.gop.broadcast(cell_flat.data(), static_cast<int>(NDIM * 2), 0);

    const size_t N   = static_cast<size_t>(num_leaf_nodes);
    const size_t vpn = kpow(k, NDIM);

    if (world.rank() != 0) {
        leaf_levels.resize(N);
        leaf_trans_flat.resize(N * NDIM);
        leaf_vals_flat.resize(N * vpn);
    }
    if (N > 0) {
        world.gop.broadcast(leaf_levels.data(),      static_cast<int>(N),        0);
        world.gop.broadcast(leaf_trans_flat.data(),  static_cast<int>(N * NDIM), 0);
        world.gop.broadcast(leaf_vals_flat.data(),   static_cast<int>(N * vpn),  0);
    }

    // ---- restore simulation cell ----

    // FunctionDefaults<NDIM>::set_cell expects a Tensor<double> of shape [NDIM][2]
    Tensor<double> cell_tensor(static_cast<long>(NDIM), 2L);
    for (std::size_t d = 0; d < NDIM; ++d) {
        cell_tensor(static_cast<long>(d), 0L) = cell_flat[d * 2    ];  // lo
        cell_tensor(static_cast<long>(d), 1L) = cell_flat[d * 2 + 1];  // hi
    }
    FunctionDefaults<NDIM>::set_cell(cell_tensor);

    // ---- construct empty function then insert leaf nodes ----

    Function<T, NDIM> f = FunctionFactory<T, NDIM>(world).k(k).empty();
    world.gop.fence();

    auto& coeffs = f.get_impl()->get_coeffs();

    for (size_t idx = 0; idx < N; ++idx) {
        Level level = static_cast<Level>(leaf_levels[idx]);

        Vector<Translation, NDIM> trans;
        for (std::size_t d = 0; d < NDIM; ++d)
            trans[d] = static_cast<Translation>(leaf_trans_flat[idx * NDIM + d]);

        Key<NDIM> key(level, trans);

        // Tensor shape [k, k, ..., k]  (NDIM times)
        long dims[NDIM];
        for (std::size_t d = 0; d < NDIM; ++d)
            dims[d] = static_cast<long>(k);
        Tensor<T> values(static_cast<long>(NDIM), dims);

        const double* src = leaf_vals_flat.data() + idx * vpn;
        T* dst = values.ptr();
        for (size_t i = 0; i < vpn; ++i)
            dst[i] = static_cast<T>(src[i]);

        // values -> scaling-function coefficients, accumulate into tree
        auto t = f.get_impl()->values2coeffs(key, values);
        coeffs.task(key, &FunctionNode<T, NDIM>::accumulate2, t, coeffs, key);
    }

    world.gop.fence();
    f.verify_tree();

    return f;
}


// ============================================================
// Explicit instantiations  (double + float, NDIM = 1, 2, 3, 6)
// ============================================================

#define INSTANTIATE(T, N) \
    template void            save_function_v1<T, N>(const Function<T, N>&, const std::string&); \
    template Function<T, N>  load_function_v1<T, N>(World&,               const std::string&);

INSTANTIATE(double, 1)
INSTANTIATE(double, 2)
INSTANTIATE(double, 3)
INSTANTIATE(double, 6)

INSTANTIATE(float, 1)
INSTANTIATE(float, 2)
INSTANTIATE(float, 3)
INSTANTIATE(float, 6)

#undef INSTANTIATE

}  // namespace hdf5
}  // namespace io
}  // namespace madness
