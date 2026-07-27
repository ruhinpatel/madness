#!/usr/bin/env python
"""Step 2 gate: compress -> reconstruct roundtrip on a real HDF5-loaded function.

Validates that pymra's two-scale compress/reconstruct are exact inverses when
applied to functions loaded from MADNESS .mad.h5 files. Because hg is orthogonal
the roundtrip should recover leaf s-coefficients to machine precision (~1e-12),
well below thresh.

Usage:
    python step2_gate.py <path>.mad.h5
"""

import sys
import numpy as np

sys.path.insert(0, str(__file__).rsplit("/", 1)[0])  # find reconstruct.py

from pymra import read_function
from pymra.twoscale import compress
from reconstruct import reconstruct


def main():
    if len(sys.argv) < 2:
        print("Usage: step2_gate.py <path>.mad.h5")
        sys.exit(1)

    h5_path = sys.argv[1]
    tree = read_function(h5_path)
    print(f"loaded  {h5_path}")
    print(f"  k={tree.k}  thresh={tree.thresh}  leaves={sum(1 for _ in tree.leaves())}")

    comp = compress(tree)
    rec  = reconstruct(comp, tree)

    errs = []
    for key, node in tree.leaves():
        rec_node = rec.nodes.get(key)
        if rec_node is None or not rec_node.has_coeff:
            print(f"FAIL: leaf {key} missing in reconstructed tree")
            sys.exit(1)
        errs.append(float(np.max(np.abs(node.s - rec_node.s))))

    max_err = max(errs)
    tol = 1e-10  # well below thresh; roundtrip error should be ~machine precision
    print(f"leaves: {len(errs)}  max coeff err: {max_err:.3e}  tol: {tol:.1e}")
    verdict = "PASS" if max_err <= tol else "FAIL"
    print(verdict)
    sys.exit(0 if verdict == "PASS" else 1)


if __name__ == "__main__":
    main()
