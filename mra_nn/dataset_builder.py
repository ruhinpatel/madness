"""MRA-NN training dataset builder  (Step 4).

For each molecule's (rho0, vnuc, rho) .mad.h5 triple, enumerates all boxes in
the rho tree at every level (internal nodes + leaves), computes input features
(s-coefficients of rho0 and vnuc at the box and its 6 face-adjacent halo boxes),
and stores targets (Δρ s-coefficients, log‖d_rho‖, refine flag).  Below-leaf
negatives (children of rho leaves that are NOT in the tree) are added as
refine=0 samples with d ≈ 0.

Gate (--gate): verifies the stored rho0_s + delta_rho reproduces the original
rho leaf coefficients exactly (machine precision), then checks ∫ρ = N.

Usage:
    python dataset_builder.py \\
        --data-dir /gpfs/projects/rjh/ruhin/mra_nn/training_data \\
        --out      /gpfs/projects/rjh/ruhin/mra_nn/training_dataset.h5 \\
        [--mols h2o nh3 ...] \\
        [--gate]

HDF5 layout:
    /attrs:  k, ndim, molecules
    /<mol>/
        rho0_s    [N, k^3]    float32  rho0 s-coeffs at box
        vnuc_s    [N, k^3]    float32  vnuc s-coeffs at box
        halo_rho0 [N, 6, k^3] float32  rho0 s-coeffs at 6 face-adjacent halos
        halo_vnuc [N, 6, k^3] float32  vnuc s-coeffs at 6 face-adjacent halos
        delta_rho [N, k^3]    float32  target: (rho - rho0) s-coeffs at box
        log_dnorm [N]         float32  target: log(||d_rho||), floor -30
        refine    [N]         int8     target: 1=needs refinement, 0=leaf/negative
        level     [N]         int8     box level n
        l_trans   [N, 3]      int32    box translation vector l
        negative  [N]         int8     1=below-leaf negative, 0=positive sample
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import h5py
import numpy as np

# find reconstruct.py in the same directory
sys.path.insert(0, str(Path(__file__).parent))

from pymra import read_function, FunctionTree
from pymra.tree import Key, Node
from pymra.twoscale import compress, node_s

# 6 face-adjacent offsets in 3D (+x, -x, +y, -y, +z, -z)
HALO_OFFSETS = [
    (1, 0, 0), (-1, 0, 0),
    (0, 1, 0), (0, -1, 0),
    (0, 0, 1), (0, 0, -1),
]
N_HALO = len(HALO_OFFSETS)


def is_valid_key(key: Key) -> bool:
    """True if all translations are in [0, 2^n) — i.e. within the simulation cell."""
    max_t = 1 << key.n
    return all(0 <= li < max_t for li in key.l)


def safe_node_s(tree: FunctionTree, key: Key, k: int, ndim: int) -> np.ndarray:
    """node_s with zero-padding for out-of-cell halo boxes."""
    if not is_valid_key(key):
        return np.zeros((k,) * ndim)
    return node_s(tree, key)


def halo_keys(key: Key):
    """Yield 6 face-adjacent neighbor Keys at the same level."""
    n, l = key.n, key.l
    for off in HALO_OFFSETS:
        yield Key(n, tuple(l[d] + off[d] for d in range(key.ndim)))


def process_molecule(mol_dir: Path) -> dict:
    """Build dataset arrays for one molecule.

    Returns a dict of numpy arrays keyed by dataset name.
    """
    rho0 = read_function(str(mol_dir / "rho0.mad.h5"))
    vnuc = read_function(str(mol_dir / "vnuc.mad.h5"))
    rho  = read_function(str(mol_dir / "rho.mad.h5"))

    k, ndim = rho.k, rho.ndim
    comp_rho = compress(rho)

    rho0_s_list    = []
    vnuc_s_list    = []
    halo_rho0_list = []
    halo_vnuc_list = []
    delta_rho_list = []
    log_dnorm_list = []
    refine_list    = []
    level_list     = []
    l_trans_list   = []
    negative_list  = []

    def _append(key: Key, rho_s: np.ndarray, log_d: float, refine: int, neg: int):
        rho0_s = node_s(rho0, key)
        vnuc_s = node_s(vnuc, key)
        h_rho0 = np.stack([safe_node_s(rho0, hk, k, ndim).ravel()
                           for hk in halo_keys(key)])          # [6, k^3]
        h_vnuc = np.stack([safe_node_s(vnuc, hk, k, ndim).ravel()
                           for hk in halo_keys(key)])
        drho   = (rho_s - rho0_s).ravel()

        rho0_s_list.append(rho0_s.ravel())
        vnuc_s_list.append(vnuc_s.ravel())
        halo_rho0_list.append(h_rho0)
        halo_vnuc_list.append(h_vnuc)
        delta_rho_list.append(drho)
        log_dnorm_list.append(log_d)
        refine_list.append(refine)
        level_list.append(key.n)
        l_trans_list.append(list(key.l))
        negative_list.append(neg)

    # --- positive samples: all nodes in the rho tree ---
    for key, node in rho.nodes.items():
        rho_s = node_s(rho, key)
        if key in comp_rho.d:
            # internal node: has d-coefficients → needs refinement
            log_d  = math.log(max(comp_rho.dnorm(key), 1e-30))
            refine = 1
        else:
            # leaf node: function is smooth here
            log_d  = -30.0
            refine = 0
        _append(key, rho_s, log_d, refine, neg=0)

    # --- below-leaf negatives: children of rho leaves ---
    for key, _ in rho.leaves():
        for child in key.children():
            rho_s = node_s(rho, child)   # refines down from parent (d=0 below leaf)
            _append(child, rho_s, log_d=-30.0, refine=0, neg=1)

    return {
        "rho0_s":    np.array(rho0_s_list,    dtype=np.float32),
        "vnuc_s":    np.array(vnuc_s_list,    dtype=np.float32),
        "halo_rho0": np.array(halo_rho0_list, dtype=np.float32),
        "halo_vnuc": np.array(halo_vnuc_list, dtype=np.float32),
        "delta_rho": np.array(delta_rho_list, dtype=np.float32),
        "log_dnorm": np.array(log_dnorm_list, dtype=np.float32),
        "refine":    np.array(refine_list,    dtype=np.int8),
        "level":     np.array(level_list,     dtype=np.int8),
        "l_trans":   np.array(l_trans_list,   dtype=np.int32),
        "negative":  np.array(negative_list,  dtype=np.int8),
    }


def write_molecule(hf: h5py.File, mol_name: str, data: dict) -> None:
    """Write one molecule's arrays to an open HDF5 file."""
    grp = hf.require_group(mol_name)
    for name, arr in data.items():
        if name in grp:
            del grp[name]
        grp.create_dataset(name, data=arr, compression="gzip", compression_opts=4)
    n = len(data["level"])
    n_pos = int(np.sum(data["negative"] == 0))
    n_neg = int(np.sum(data["negative"] == 1))
    n_ref = int(np.sum(data["refine"] == 1))
    print(f"  {mol_name}: {n} samples  ({n_pos} positive [{n_ref} refine], {n_neg} negatives)")


def gate_check(mol_name: str, data: dict, mol_dir: Path) -> bool:
    """Verify rho0_s + delta_rho reproduces the original rho leaf coefficients,
    then check ∫ρ = N via the original tree's integral()."""
    rho0 = read_function(str(mol_dir / "rho0.mad.h5"))
    rho  = read_function(str(mol_dir / "rho.mad.h5"))
    k, ndim = rho.k, rho.ndim

    # Leaf entries: refine=0 AND negative=0
    mask = (data["refine"] == 0) & (data["negative"] == 0)
    idx  = np.where(mask)[0]

    max_err = 0.0
    for i in idx:
        n_lev = int(data["level"][i])
        l     = tuple(int(x) for x in data["l_trans"][i])
        key   = Key(n_lev, l)
        # reconstruct rho s-coefficients from dataset
        rho_s_from_data = (data["rho0_s"][i] + data["delta_rho"][i]).reshape((k,) * ndim)
        rho_s_from_tree = node_s(rho, key)
        err = float(np.max(np.abs(rho_s_from_data - rho_s_from_tree)))
        if err > max_err:
            max_err = err

    tol = 1e-5  # float32 storage rounds to ~1e-7, accumulated across sums
    coeff_ok = max_err < tol
    print(f"  gate [{mol_name}] leaf coeff max err: {max_err:.3e}  tol: {tol:.1e}  "
          f"{'OK' if coeff_ok else 'FAIL'}")

    # integral check: rebuild rho tree from leaf entries and integrate
    rho_tree = FunctionTree(k=k, ndim=ndim, cell=rho.cell.copy(), thresh=rho.thresh,
                             initial_level=rho.initial_level)
    for i in idx:
        n_lev = int(data["level"][i])
        l     = tuple(int(x) for x in data["l_trans"][i])
        key   = Key(n_lev, l)
        s     = (data["rho0_s"][i] + data["delta_rho"][i]).reshape((k,) * ndim)
        rho_tree.nodes[key] = Node(s=s.astype(float))
    # internal node stubs: add has_children nodes so the tree is navigable
    for i in np.where(~mask & (data["negative"] == 0))[0]:
        n_lev = int(data["level"][i])
        l     = tuple(int(x) for x in data["l_trans"][i])
        key   = Key(n_lev, l)
        if key not in rho_tree.nodes:
            rho_tree.nodes[key] = Node(has_children=True)

    integral   = rho_tree.integral()
    N_expected = rho.integral()
    int_err    = abs(integral - N_expected)
    int_ok     = int_err < 1e-3
    print(f"  gate [{mol_name}] ∫ρ = {integral:.6f}  expected: {N_expected:.6f}  "
          f"err: {int_err:.3e}  {'OK' if int_ok else 'FAIL'}")

    ok = coeff_ok and int_ok
    print(f"  gate [{mol_name}] {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    parser = argparse.ArgumentParser(description="MRA-NN training dataset builder")
    parser.add_argument("--data-dir", required=True,
                        help="Path to training_data/ directory")
    parser.add_argument("--out", required=True,
                        help="Output HDF5 path")
    parser.add_argument("--mols", nargs="*", default=None,
                        help="Molecules to process (default: all subdirectories)")
    parser.add_argument("--gate", action="store_true",
                        help="Run gate check after building (on first molecule)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if args.mols:
        molecules = args.mols
    else:
        molecules = sorted(p.name for p in data_dir.iterdir()
                           if p.is_dir() and (p / "rho.mad.h5").exists())

    print(f"Building dataset: {len(molecules)} molecules → {args.out}")

    first_mol_data = None
    with h5py.File(args.out, "w") as hf:
        hf.attrs["molecules"] = molecules

        for mol in molecules:
            mol_dir = data_dir / mol
            print(f"Processing {mol} ...")
            data = process_molecule(mol_dir)
            write_molecule(hf, mol, data)
            if first_mol_data is None:
                first_mol_data = (mol, data, mol_dir)

        # store k and ndim from the first molecule
        sample_tree = read_function(str(data_dir / molecules[0] / "rho.mad.h5"))
        hf.attrs["k"]    = sample_tree.k
        hf.attrs["ndim"] = sample_tree.ndim

    print(f"\nDataset written to {args.out}")

    if args.gate and first_mol_data is not None:
        mol, data, mol_dir = first_mol_data
        print(f"\nRunning gate check on {mol} ...")
        ok = gate_check(mol, data, mol_dir)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
