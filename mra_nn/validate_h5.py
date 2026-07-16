#!/usr/bin/env python3
"""Validate .mad.h5 training data files for MRA-NN."""

import sys
from pathlib import Path

import h5py

if len(sys.argv) > 1:
    mol_dir_arg = Path(sys.argv[1]).resolve()
    DATA_DIR = mol_dir_arg.parent
    MOLECULES = [mol_dir_arg.name]
else:
    DATA_DIR = Path("/gpfs/projects/rjh/ruhin/mra_nn/training_data")
    MOLECULES = ["h2o", "nh3", "ch4", "co2", "hf"]
FILES_PER_MOL = ["rho0.mad.h5", "vnuc.mad.h5", "rho.mad.h5"]

# Expected meta attributes written by save_function_hdf5
META_ATTRS = ["schema", "k", "thresh", "ndim", "tree_state",
              "initial_level", "truncate_mode", "autorefine",
              "n_nodes", "n_coeff_nodes", "cell"]

errors = 0

for mol in MOLECULES:
    mol_dir = DATA_DIR / mol
    for fname in FILES_PER_MOL:
        fpath = mol_dir / fname
        label = f"{mol}/{fname}"

        if not fpath.exists():
            print(f"FAIL  {label}: file missing")
            errors += 1
            continue

        try:
            with h5py.File(fpath, "r") as f:
                # Check /meta group
                if "meta" not in f:
                    print(f"FAIL  {label}: /meta group missing")
                    errors += 1
                    continue

                meta = f["meta"]
                missing = [a for a in META_ATTRS if a not in meta.attrs]
                if missing:
                    print(f"FAIL  {label}: /meta missing attrs: {missing}")
                    errors += 1
                    continue

                k = int(meta.attrs["k"])
                ndim = int(meta.attrs["ndim"])
                n_nodes = int(meta.attrs["n_nodes"])
                n_coeff = int(meta.attrs["n_coeff_nodes"])

                # Check /keys dataset
                if "keys" not in f:
                    print(f"FAIL  {label}: /keys dataset missing")
                    errors += 1
                    continue
                keys_shape = f["keys"].shape
                expect_keys = (n_nodes, 3 + ndim)
                if keys_shape != expect_keys:
                    print(f"FAIL  {label}: /keys shape {keys_shape} != expected {expect_keys}")
                    errors += 1
                    continue

                # Check /coeffs dataset
                if "coeffs" not in f:
                    print(f"FAIL  {label}: /coeffs dataset missing")
                    errors += 1
                    continue
                coeffs_shape = f["coeffs"].shape
                npts = k ** ndim
                expect_coeffs = (n_coeff, npts)
                if coeffs_shape != expect_coeffs:
                    print(f"FAIL  {label}: /coeffs shape {coeffs_shape} != expected {expect_coeffs}")
                    errors += 1
                    continue

                size_mb = fpath.stat().st_size / 1e6
                print(f"  OK  {label}: k={k} ndim={ndim} nodes={n_nodes} "
                      f"coeff_nodes={n_coeff} ({size_mb:.1f} MB)")

        except Exception as e:
            print(f"FAIL  {label}: {e}")
            errors += 1

print()
if errors:
    print(f"{errors} error(s) found.")
    sys.exit(1)
else:
    n_expected = len(MOLECULES) * len(FILES_PER_MOL)
    print(f"All {n_expected} files validated successfully.")
