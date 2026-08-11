"""Tree-walk inference for MRA-NN.

Two inference modes:

**Multi-task** (refine head available):
  Walk the tree top-down, using the refine_logit head to decide structure.

**Single-task** (no refine head):
  Copy rho0's tree topology, replace leaf s-coefficients with model
  predictions.  After prediction, run compress->reconstruct to ensure
  the two-scale relation is satisfied throughout the tree.

Post-processing: scale all leaf coefficients so integral(rho) = N.

Usage:
    python predict.py --checkpoint best.pt --rho0 rho0.mad.h5 \\
                      --vnuc vnuc.mad.h5 --n-electrons 10 --out predicted.mad.h5
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from typing import List

import numpy as np
import torch
import yaml

from pymra import FunctionTree, read_function, write_function
from pymra.tree import Key, Node
from pymra.twoscale import compress, node_s, reconstruct, refine as twoscale_refine

from mra_nn.model import MRANet, build_model


# 6 face-adjacent offsets in 3D: +x, -x, +y, -y, +z, -z
HALO_OFFSETS = [
    (1, 0, 0), (-1, 0, 0),
    (0, 1, 0), (0, -1, 0),
    (0, 0, 1), (0, 0, -1),
]


def _is_valid_key(key: Key) -> bool:
    max_t = 1 << key.n
    return all(0 <= li < max_t for li in key.l)


def _safe_node_s(tree: FunctionTree, key: Key) -> np.ndarray:
    """node_s with zero-padding for out-of-cell halo boxes."""
    if not _is_valid_key(key):
        return np.zeros((tree.k,) * tree.ndim)
    return node_s(tree, key)


def _halo_keys(key: Key) -> List[Key]:
    return [
        Key(key.n, tuple(key.l[d] + off[d] for d in range(key.ndim)))
        for off in HALO_OFFSETS
    ]


def _extract_features(
    keys: List[Key],
    rho0_tree: FunctionTree,
    vnuc_tree: FunctionTree,
    device: torch.device,
    include_parent: bool = False,
) -> dict:
    """Extract model input features for a batch of keys."""
    k = rho0_tree.k
    ndim = rho0_tree.ndim

    rho0_s_list = []
    vnuc_s_list = []
    halo_rho0_list = []
    halo_vnuc_list = []
    level_list = []
    parent_rho0_list = []
    parent_vnuc_list = []

    for key in keys:
        rho0_s_list.append(node_s(rho0_tree, key).ravel())
        vnuc_s_list.append(node_s(vnuc_tree, key).ravel())

        h_rho0 = np.stack([_safe_node_s(rho0_tree, hk).ravel()
                           for hk in _halo_keys(key)])
        h_vnuc = np.stack([_safe_node_s(vnuc_tree, hk).ravel()
                           for hk in _halo_keys(key)])
        halo_rho0_list.append(h_rho0)
        halo_vnuc_list.append(h_vnuc)
        level_list.append(key.n)

        if include_parent:
            if key.n > 0:
                parent_key = key.parent()
                parent_rho0_list.append(node_s(rho0_tree, parent_key).ravel())
                parent_vnuc_list.append(node_s(vnuc_tree, parent_key).ravel())
            else:
                parent_rho0_list.append(np.zeros(k ** ndim, dtype=np.float64))
                parent_vnuc_list.append(np.zeros(k ** ndim, dtype=np.float64))

    result = {
        "rho0_s": torch.from_numpy(np.array(rho0_s_list, dtype=np.float32)).to(device),
        "vnuc_s": torch.from_numpy(np.array(vnuc_s_list, dtype=np.float32)).to(device),
        "halo_rho0": torch.from_numpy(np.array(halo_rho0_list, dtype=np.float32)).to(device),
        "halo_vnuc": torch.from_numpy(np.array(halo_vnuc_list, dtype=np.float32)).to(device),
        "level": torch.tensor(level_list, dtype=torch.long).to(device),
    }
    if include_parent:
        result["parent_rho0_s"] = torch.from_numpy(
            np.array(parent_rho0_list, dtype=np.float32)
        ).to(device)
        result["parent_vnuc_s"] = torch.from_numpy(
            np.array(parent_vnuc_list, dtype=np.float32)
        ).to(device)
    return result


def _ensure_children_exist(
    tree: FunctionTree, parent_key: Key
) -> List[Key]:
    """Ensure rho0/vnuc have s-coefficients at parent_key's children.

    If the tree doesn't go that deep, refine the parent's coefficients down.
    Returns list of child keys.
    """
    children = list(parent_key.children())
    node = tree.nodes.get(parent_key)
    if node is None:
        # This shouldn't happen in normal use, but handle gracefully
        return children

    # Check if children already exist
    if all(ck in tree.nodes and tree.nodes[ck].has_coeff for ck in children):
        return children

    # Need to refine parent down
    parent_s = node_s(tree, parent_key)
    child_coeffs = twoscale_refine(parent_s)

    for bits, child_s in child_coeffs.items():
        child_key = Key(
            parent_key.n + 1,
            tuple(2 * parent_key.l[d] + bits[d] for d in range(tree.ndim)),
        )
        if child_key not in tree.nodes or not tree.nodes[child_key].has_coeff:
            tree.nodes[child_key] = Node(s=child_s)

    return children


def _normalize_integral(tree: FunctionTree, n_electrons: int) -> None:
    """Scale all leaf coefficients so integral(rho) = n_electrons."""
    integral = tree.integral()
    if abs(integral) > 1e-10:
        scale = n_electrons / integral
        for _, node in tree.leaves():
            node.s = node.s * scale


def _compress_reconstruct(tree: FunctionTree) -> FunctionTree:
    """Run compress->reconstruct to enforce two-scale consistency."""
    comp = compress(tree)
    return reconstruct(comp)


@torch.no_grad()
def predict_density_simple(
    model: MRANet,
    rho0_path: str,
    vnuc_path: str,
    n_electrons: int,
    device: torch.device,
    batch_size: int = 4096,
    use_model_levels: set[int] | None = None,
) -> FunctionTree:
    """Predict density using rho0's tree topology (single-task mode).

    Walks rho0's existing leaf set, replaces each leaf's s-coefficients
    with the model's prediction, then runs compress->reconstruct to
    enforce two-scale consistency.

    Parameters
    ----------
    model : trained single-task MRANet
    rho0_path : path to rho0.mad.h5
    vnuc_path : path to vnuc.mad.h5
    n_electrons : number of electrons (for integral normalization)
    device : torch device
    batch_size : number of leaves to process per forward pass
    use_model_levels : set of tree levels at which to use the model prediction.
        At all other levels, rho0's s-coefficients are used unchanged.
        Default None means use the model at all levels (backward compatible).

    Returns
    -------
    FunctionTree with predicted density coefficients, integral-normalized.
    """
    model.eval()
    rho0_tree = read_function(rho0_path)
    vnuc_tree = read_function(vnuc_path)

    k = rho0_tree.k
    ndim = rho0_tree.ndim

    # Collect rho0's leaf keys
    leaf_keys = [key for key, _ in rho0_tree.leaves()]

    # Build output tree with same internal structure as rho0
    predicted_tree = FunctionTree(
        k=k, ndim=ndim,
        cell=rho0_tree.cell.copy(),
        thresh=rho0_tree.thresh,
        initial_level=rho0_tree.initial_level,
    )
    # Copy internal (non-leaf) nodes
    for key, node in rho0_tree.nodes.items():
        if not node.has_coeff:
            predicted_tree.nodes[key] = Node(has_children=True)

    # Detect if model uses parent features
    use_parent = getattr(model, 'use_parent_features', False)

    # Predict in batches
    for start in range(0, len(leaf_keys), batch_size):
        batch_keys = leaf_keys[start : start + batch_size]
        features = _extract_features(
            batch_keys, rho0_tree, vnuc_tree, device,
            include_parent=use_parent,
        )
        forward_args = [
            features["rho0_s"], features["vnuc_s"],
            features["halo_rho0"], features["halo_vnuc"],
            features["level"],
        ]
        if use_parent:
            forward_args.extend([features["parent_rho0_s"], features["parent_vnuc_s"]])
        rho_s, _, _ = model(*forward_args)
        rho_s_np = rho_s.cpu().numpy()
        for i, key in enumerate(batch_keys):
            if use_model_levels is not None and key.n not in use_model_levels:
                # Outside effective range — use rho0 as-is
                predicted_tree.nodes[key] = Node(
                    s=node_s(rho0_tree, key).copy()
                )
            else:
                predicted_tree.nodes[key] = Node(
                    s=rho_s_np[i].reshape((k,) * ndim).astype(np.float64)
                )

    # Compress->reconstruct for two-scale consistency
    predicted_tree = _compress_reconstruct(predicted_tree)

    _normalize_integral(predicted_tree, n_electrons)
    return predicted_tree


@torch.no_grad()
def predict_density(
    model: MRANet,
    rho0_path: str,
    vnuc_path: str,
    n_electrons: int,
    device: torch.device,
    refine_threshold: float = 0.5,
    max_level: int = 18,
) -> FunctionTree:
    """Predict density via top-down tree walk (multi-task mode).

    Requires a model with a refine_logit head. For single-task models,
    use predict_density_simple instead.

    Parameters
    ----------
    model : trained MRANet (multi-task, with refine head)
    rho0_path : path to rho0.mad.h5
    vnuc_path : path to vnuc.mad.h5
    n_electrons : number of electrons (for integral normalization)
    device : torch device
    refine_threshold : probability threshold for refinement decision
    max_level : maximum tree depth (safety limit)

    Returns
    -------
    FunctionTree with predicted density coefficients, integral-normalized.
    """
    model.eval()
    rho0_tree = read_function(rho0_path)
    vnuc_tree = read_function(vnuc_path)

    k = rho0_tree.k
    ndim = rho0_tree.ndim

    predicted_tree = FunctionTree(
        k=k, ndim=ndim,
        cell=rho0_tree.cell.copy(),
        thresh=rho0_tree.thresh,
        initial_level=rho0_tree.initial_level,
    )

    # Start at root
    root_key = Key(0, (0,) * ndim)
    current_level_keys = [root_key]

    while current_level_keys:
        features = _extract_features(
            current_level_keys, rho0_tree, vnuc_tree, device
        )
        rho_s, log_dnorm, refine_logit = model(
            features["rho0_s"], features["vnuc_s"],
            features["halo_rho0"], features["halo_vnuc"],
            features["level"],
        )

        refine_prob = torch.sigmoid(refine_logit).cpu().numpy()
        rho_s_np = rho_s.cpu().numpy()

        next_level_keys = []
        for i, key in enumerate(current_level_keys):
            if refine_prob[i] > refine_threshold and key.n < max_level:
                # Internal node — refine
                predicted_tree.nodes[key] = Node(has_children=True)
                # Ensure rho0/vnuc have children for feature extraction
                _ensure_children_exist(rho0_tree, key)
                _ensure_children_exist(vnuc_tree, key)
                next_level_keys.extend(key.children())
            else:
                # Leaf — write predicted coefficients directly
                pred_s = rho_s_np[i]
                predicted_tree.nodes[key] = Node(
                    s=pred_s.reshape((k,) * ndim).astype(np.float64)
                )

        current_level_keys = next_level_keys

    _normalize_integral(predicted_tree, n_electrons)
    return predicted_tree


def main():
    parser = argparse.ArgumentParser(description="MRA-NN inference")
    parser.add_argument("--checkpoint", required=True, help="Path to best.pt")
    parser.add_argument("--rho0", required=True, help="Path to rho0.mad.h5")
    parser.add_argument("--vnuc", required=True, help="Path to vnuc.mad.h5")
    parser.add_argument("--n-electrons", type=int, required=True)
    parser.add_argument("--out", required=True, help="Output .mad.h5 path")
    parser.add_argument("--refine-threshold", type=float, default=0.5)
    parser.add_argument(
        "--use-model-levels", type=str, default=None,
        help="Comma-separated levels where model predictions are used (e.g. '10,11,12,13,14'). "
             "At other levels, rho0 is used unchanged. Default: use model at all levels.",
    )
    args = parser.parse_args()

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    single_task = cfg["model"].get("single_task", False)
    print(f"Loaded model from {args.checkpoint} (epoch {ckpt['epoch']})")
    print(f"Mode: {'single-task' if single_task else 'multi-task'}")
    print(f"Predicting density for rho0={args.rho0}, vnuc={args.vnuc}")

    use_model_levels = None
    if args.use_model_levels:
        use_model_levels = set(int(x) for x in args.use_model_levels.split(","))
        print(f"Level clamping: using model at levels {sorted(use_model_levels)}, rho0 elsewhere")

    if single_task:
        tree = predict_density_simple(
            model, args.rho0, args.vnuc,
            n_electrons=args.n_electrons,
            device=device,
            use_model_levels=use_model_levels,
        )
    else:
        tree = predict_density(
            model, args.rho0, args.vnuc,
            n_electrons=args.n_electrons,
            device=device,
            refine_threshold=args.refine_threshold,
        )

    n_leaves = sum(1 for _ in tree.leaves())
    integral = tree.integral()
    print(f"Predicted tree: {n_leaves} leaves, integral={integral:.6f}")

    write_function(tree, args.out)
    print(f"Written to {args.out}")


if __name__ == "__main__":
    main()
