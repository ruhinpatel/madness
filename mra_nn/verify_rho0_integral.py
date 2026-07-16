"""Verify that ∫ρ₀ = N for each molecule's promolecular density using pymra."""

import sys
sys.path.insert(0, "/gpfs/projects/rjh/adrian/pymra/src")
from pymra import read_function

MOLECULES = {
    "h2o": 10,
    "nh3": 10,
    "ch4": 10,
    "co2": 22,
    "hf": 10,
}

DATA_DIR = "/gpfs/projects/rjh/ruhin/mra_nn/training_data"

all_pass = True
for mol, expected_N in MOLECULES.items():
    path = f"{DATA_DIR}/{mol}/rho0.mad.h5"
    tree = read_function(path)
    integral = tree.integral()
    diff = abs(integral - expected_N)
    status = "PASS" if diff < 0.1 else "FAIL"
    if status == "FAIL":
        all_pass = False
    print(f"{mol:4s}  ∫ρ₀ = {integral:8.4f}  expected = {expected_N:2d}  diff = {diff:.4f}  {status}")

sys.exit(0 if all_pass else 1)
