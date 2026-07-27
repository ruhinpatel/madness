"""reconstruct() — inverse of pymra.twoscale.compress().

Proposed addition to pymra.twoscale; lives here until Adrian merges it.

Usage:
    from reconstruct import reconstruct
    comp = compress(tree)
    rec  = reconstruct(comp, tree)
    # rec has the same leaf s-coefficients as tree (to machine precision)
"""

from __future__ import annotations

from pymra.tree import FunctionTree, Key, Node
from pymra.twoscale import Compressed, unfilter_nd


def _child_slices(bits, k):
    return tuple(slice(b * k, (b + 1) * k) for b in bits)


def _reconstruct_walk(
    comp: Compressed,
    orig_tree: FunctionTree,
    key: Key,
    s,
    new_tree: FunctionTree,
) -> None:
    node = orig_tree.nodes.get(key)
    if node is None:
        return
    if node.has_coeff:
        # leaf: store the accumulated scaling coefficients
        new_tree.nodes[key] = Node(s=s.copy())
        return
    # internal node: insert s into the s-corner of the stored d block,
    # apply unfilter to recover the children's scaling coefficients
    k = comp.k
    ndim = comp.ndim
    sd = comp.d[key].copy()             # d-only block (s-corner is zero)
    sd[(slice(0, k),) * ndim] = s      # insert parent s
    u = unfilter_nd(sd)                 # children gather: (2k,)*ndim
    new_tree.nodes[key] = Node(has_children=True)
    for child in key.children():
        bits = tuple(child.l[d] - 2 * key.l[d] for d in range(ndim))
        _reconstruct_walk(
            comp, orig_tree, child, u[_child_slices(bits, k)].copy(), new_tree
        )


def reconstruct(comp: Compressed, tree: FunctionTree) -> FunctionTree:
    """Top-down unfilter sweep: inverse of compress().

    Takes a Compressed result and the original FunctionTree (used for tree
    structure / leaf positions) and returns a new FunctionTree with leaf
    s-coefficients rebuilt from comp.s0 and the wavelet blocks in comp.d.

    The roundtrip reconstruct(compress(tree), tree) recovers the original
    leaf coefficients to machine precision because hg is orthogonal.
    """
    new_tree = FunctionTree(
        k=tree.k,
        ndim=tree.ndim,
        cell=tree.cell.copy(),
        thresh=tree.thresh,
        initial_level=tree.initial_level,
        truncate_mode=tree.truncate_mode,
        autorefine=tree.autorefine,
    )
    _reconstruct_walk(comp, tree, Key(0, (0,) * tree.ndim), comp.s0.copy(), new_tree)
    return new_tree
