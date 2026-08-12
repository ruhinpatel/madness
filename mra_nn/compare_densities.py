"""Compare ρ₀, ρ_ML, and ρ_converged densities for SCF initial-guess analysis.

Computes per-leaf and global error metrics between density pairs to assess
whether the ML-predicted density is closer to the converged SCF density
than the default promolecular guess (ρ₀).

Usage:
    python compare_densities.py \
        --rho0 training_data/ch3oh/rho0.mad.h5 \
        --rho-conv training_data/ch3oh/rho.mad.h5 \
        --rho-ml scf_test/ch3oh/rhoML.mad.h5
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict

import numpy as np

from pymra import FunctionTree, read_function
from pymra.tree import Key, Node
from pymra.twoscale import compress, node_s, reconstruct, refine as twoscale_refine


def _refine_to_key(tree: FunctionTree, target_key: Key) -> np.ndarray:
    """Get s-coefficients at target_key, refining ancestors if needed.

    If target_key is a leaf or internal node, return its s-coeffs directly.
    If it lives below a leaf in the tree, refine the ancestor leaf down
    to the target level and return the resulting coefficients.
    """
    # Walk up from target_key until we find a node that exists in the tree
    key = target_key
    path = []
    while key not in tree.nodes:
        path.append(key)
        if key.n == 0:
            # Target is outside the tree entirely — return zeros
            return np.zeros((tree.k,) * tree.ndim)
        parent_n = key.n - 1
        parent_l = tuple(li // 2 for li in key.l)
        key = Key(parent_n, parent_l)

    node = tree.nodes[key]
    if not node.has_coeff:
        # Internal node without coefficients — shouldn't happen for
        # well-formed trees, but return zeros as fallback
        return np.zeros((tree.k,) * tree.ndim)

    # Now refine down through path (reversed, so top-down)
    s = node.s
    for child_key in reversed(path):
        child_coeffs = twoscale_refine(s)
        # Determine which child we need
        parent_n = child_key.n - 1
        bits = tuple(child_key.l[d] % 2 for d in range(tree.ndim))
        s = child_coeffs[bits]

    return s


def _common_leaf_set(tree_a: FunctionTree, tree_b: FunctionTree) -> list[Key]:
    """Find the finest common leaf decomposition of two trees.

    For each spatial region, use the deeper tree's leaves. Where tree A
    is deeper, refine tree B's coefficients down (and vice versa).
    Returns a sorted list of keys forming the common leaf set.
    """
    leaves_a = {key for key, _ in tree_a.leaves()}
    leaves_b = {key for key, _ in tree_b.leaves()}

    common = set()

    for key in leaves_a:
        # Check if this leaf overlaps with tree_b's leaves
        # If key is a leaf in both trees at the same level, use it directly
        if key in leaves_b:
            common.add(key)
        elif any(key.n > bkey.n for bkey in leaves_b):
            # key is finer than some leaves in B — use key (refine B down)
            common.add(key)
        else:
            # key might be coarser than B's leaves in this region
            # Check if B has finer leaves under this key
            children_in_b = [bk for bk in leaves_b
                             if bk.n > key.n and _is_ancestor(key, bk)]
            if children_in_b:
                common.update(children_in_b)
            else:
                common.add(key)

    # Also add any leaves from B that aren't ancestors/descendants of A leaves
    for key in leaves_b:
        if key not in common:
            children_in_a = [ak for ak in leaves_a
                             if ak.n > key.n and _is_ancestor(key, ak)]
            if children_in_a:
                # A's finer leaves should already be in common
                pass
            else:
                ancestors_in_a = [ak for ak in leaves_a
                                  if ak.n < key.n and _is_ancestor(ak, key)]
                if ancestors_in_a:
                    common.add(key)

    return sorted(common, key=lambda k: (k.n, k.l))


def _is_ancestor(ancestor: Key, descendant: Key) -> bool:
    """Check if ancestor is an ancestor of descendant in the octree."""
    if ancestor.n >= descendant.n:
        return False
    level_diff = descendant.n - ancestor.n
    for d in range(len(ancestor.l)):
        if descendant.l[d] >> level_diff != ancestor.l[d]:
            return False
    return True


def compute_difference_metrics(
    tree_a: FunctionTree,
    tree_b: FunctionTree,
    label_a: str = "A",
    label_b: str = "B",
) -> dict:
    """Compute ||tree_a - tree_b|| on their common leaf set.

    Returns dict with L2 norm, relative error, max leaf error, etc.
    """
    common_keys = _common_leaf_set(tree_a, tree_b)

    sq_diff_sum = 0.0
    sq_a_sum = 0.0
    sq_b_sum = 0.0
    max_leaf_err = 0.0
    max_leaf_key = None
    n_leaves = len(common_keys)

    level_errors = defaultdict(float)
    level_counts = defaultdict(int)

    for key in common_keys:
        s_a = _refine_to_key(tree_a, key)
        s_b = _refine_to_key(tree_b, key)
        diff = s_a - s_b
        sq_diff = np.sum(diff ** 2)
        sq_diff_sum += sq_diff
        sq_a_sum += np.sum(s_a ** 2)
        sq_b_sum += np.sum(s_b ** 2)

        leaf_err = np.sqrt(sq_diff)
        if leaf_err > max_leaf_err:
            max_leaf_err = leaf_err
            max_leaf_key = key

        level_errors[key.n] += sq_diff
        level_counts[key.n] += 1

    l2_diff = np.sqrt(sq_diff_sum)
    l2_a = np.sqrt(sq_a_sum)
    l2_b = np.sqrt(sq_b_sum)
    rel_err = l2_diff / l2_a if l2_a > 0 else float("inf")

    return {
        "l2_diff": l2_diff,
        "l2_a": l2_a,
        "l2_b": l2_b,
        "relative_error": rel_err,
        "max_leaf_error": max_leaf_err,
        "max_leaf_key": max_leaf_key,
        "n_common_leaves": n_leaves,
        "level_errors": {n: np.sqrt(level_errors[n]) for n in sorted(level_errors)},
        "level_counts": {n: level_counts[n] for n in sorted(level_counts)},
    }


def compute_dipole(tree: FunctionTree) -> np.ndarray:
    """Compute electronic dipole integral(r * rho(r) dr) in atomic units.

    Uses Gauss-Legendre quadrature on each leaf cell. The MRA basis
    functions are orthonormal on [0,1] in simulation coordinates;
    the physical integral picks up a sqrt(cell_volume) factor (same
    as pymra's integral() method).
    """
    k = tree.k
    ndim = tree.ndim
    cell = tree.cell

    # Gauss-Legendre nodes and weights on [0, 1]
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(k)
    gl_nodes = 0.5 * (gl_nodes + 1.0)  # shift to [0, 1]
    gl_weights = 0.5 * gl_weights

    # Build k^3 quadrature grid on [0,1]^3
    xx = np.array(np.meshgrid(*[gl_nodes] * ndim, indexing="ij"))  # (3, k, k, k)
    ww = np.ones((k,) * ndim)
    for d in range(ndim):
        shape = [1] * ndim
        shape[d] = k
        ww = ww * gl_weights.reshape(shape)

    # Legendre polynomials evaluated at quadrature points: (k, k) matrix
    # P[i, j] = phi_i(x_j) where phi_i is the i-th scaled Legendre polynomial
    from numpy.polynomial.legendre import legval
    P = np.zeros((k, k))
    for i in range(k):
        c = np.zeros(i + 1)
        c[i] = 1.0
        P[i, :] = legval(2 * gl_nodes - 1, c) * np.sqrt(2 * i + 1)

    dipole = np.zeros(ndim)
    cell_widths = tree.cell_widths

    for key, node in tree.leaves():
        n = key.n
        s = node.s  # (k, k, k)
        box_size = 2.0 ** (-n)

        # Evaluate f_local(u) = sum_ijk s_ijk phi_i(u_x) phi_j(u_y) phi_k(u_z)
        # then f_sim(u) = 2^{3n/2} * f_local(u)
        scale = 2.0 ** (n * ndim / 2.0)
        f_vals = np.einsum("ijk,ia,jb,kc->abc", s, P, P, P) * scale

        # Physical coordinates: x_d = cell[d,0] + cell_width[d] * (l_d + u_d) / 2^n
        for d in range(ndim):
            x_phys = cell[d, 0] + cell_widths[d] * (key.l[d] + xx[d]) * box_size
            dipole[d] += np.sum(x_phys * f_vals * ww) * box_size ** ndim

    # The MRA integral picks up sqrt(cell_volume), not cell_volume
    dipole *= np.sqrt(tree.cell_volume)
    return dipole


def main():
    parser = argparse.ArgumentParser(
        description="Compare ρ₀, ρ_ML, and ρ_converged densities"
    )
    parser.add_argument("--rho0", required=True, help="Path to rho0.mad.h5")
    parser.add_argument("--rho-conv", required=True, help="Path to rho.mad.h5 (converged)")
    parser.add_argument("--rho-ml", required=True, help="Path to rhoML.mad.h5")
    args = parser.parse_args()

    print("Loading densities...")
    rho0 = read_function(args.rho0)
    rho_conv = read_function(args.rho_conv)
    rho_ml = read_function(args.rho_ml)

    # Basic stats
    print("\n" + "=" * 60)
    print("DENSITY STATISTICS")
    print("=" * 60)
    for name, tree in [("ρ₀ (promolecular)", rho0),
                       ("ρ_conv (SCF converged)", rho_conv),
                       ("ρ_ML (predicted)", rho_ml)]:
        n_leaves = sum(1 for _ in tree.leaves())
        print(f"\n  {name}:")
        print(f"    Leaves:    {n_leaves}")
        print(f"    Integral:  {tree.integral():.8f}")
        print(f"    L2 norm:   {tree.norm2():.8f}")

    # Pairwise differences
    print("\n" + "=" * 60)
    print("PAIRWISE DIFFERENCE NORMS")
    print("=" * 60)

    pairs = [
        ("ρ₀", "ρ_conv", rho0, rho_conv),
        ("ρ_ML", "ρ_conv", rho_ml, rho_conv),
        ("ρ₀", "ρ_ML", rho0, rho_ml),
    ]

    results = {}
    for label_a, label_b, tree_a, tree_b in pairs:
        print(f"\n  {label_a} vs {label_b}:")
        m = compute_difference_metrics(tree_a, tree_b, label_a, label_b)
        results[(label_a, label_b)] = m
        print(f"    ||{label_a} - {label_b}||₂ = {m['l2_diff']:.8f}")
        print(f"    Relative error:     {m['relative_error']:.6f} ({m['relative_error']*100:.2f}%)")
        print(f"    Common leaves:      {m['n_common_leaves']}")
        print(f"    Max leaf error:     {m['max_leaf_error']:.6f} at key n={m['max_leaf_key'].n}, l={m['max_leaf_key'].l}")
        print(f"    Error by level:")
        for n in sorted(m["level_errors"]):
            print(f"      level {n:2d}: {m['level_errors'][n]:.8f}  ({m['level_counts'][n]} leaves)")

    # Key comparison
    print("\n" + "=" * 60)
    print("KEY COMPARISON: Is ρ_ML closer to ρ_conv than ρ₀?")
    print("=" * 60)
    d_rho0 = results[("ρ₀", "ρ_conv")]["l2_diff"]
    d_rhoML = results[("ρ_ML", "ρ_conv")]["l2_diff"]
    ratio = d_rhoML / d_rho0 if d_rho0 > 0 else float("inf")
    print(f"\n  ||ρ₀  - ρ_conv||₂ = {d_rho0:.8f}")
    print(f"  ||ρ_ML - ρ_conv||₂ = {d_rhoML:.8f}")
    print(f"  Ratio (ML/ρ₀):       {ratio:.4f}")
    if ratio < 1.0:
        print(f"  → ρ_ML is {(1-ratio)*100:.1f}% CLOSER to ρ_conv than ρ₀  ✓")
    else:
        print(f"  → ρ_ML is {(ratio-1)*100:.1f}% FARTHER from ρ_conv than ρ₀  ✗")

    # Dipole moments
    print("\n" + "=" * 60)
    print("DIPOLE MOMENTS (a.u.)")
    print("=" * 60)
    for name, tree in [("ρ₀", rho0), ("ρ_conv", rho_conv), ("ρ_ML", rho_ml)]:
        d = compute_dipole(tree)
        mag = np.linalg.norm(d)
        print(f"\n  {name}:  x={d[0]:+.6f}  y={d[1]:+.6f}  z={d[2]:+.6f}  |d|={mag:.6f}")


if __name__ == "__main__":
    main()
