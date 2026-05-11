"""Read MADNESS JSON output files back into structured objects.

Supports two output formats:
1. MADQC ``*.calc_info.json`` — structured task-based output.
2. Legacy ``output.json`` / ``outputs.json`` — flat moldft + response data.

Mirrors the daltonproject pattern of loading results from completed
calculations, rehydrating them into the same object types used for input.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .calculation_parameters import CalculationParameters
from .molecule import Molecule
from .response_parameters import ResponseParameters


class CalculationResult:
    """Container for parsed MADNESS calculation output.

    Attributes
    ----------
    molecule : Molecule or None
        The molecular geometry from the output.
    calc_params : CalculationParameters or None
        Reconstructed calculation parameters.
    response_params : ResponseParameters or None
        Reconstructed response parameters.
    energy : float or None
        Ground-state energy in Hartree.
    alpha : dict or None
        Polarizability data ``{"omega": [...], "ij": [...], "alpha": [...]}``.
        The flat list has ``n_freqs * 9`` entries (9 tensor components per
        frequency point).
    beta : dict or None
        Hyperpolarizability data.
    raman : dict or None
        Raman spectra data from MADQC ``calc_info.json``.  Contains keys
        ``polarizability_derivatives``, ``polarizability_derivatives_normal_modes``,
        ``polarization_frequencies``, ``raman_spectra``, and
        ``vibrational_frequencies`` when present.
    raw : dict
        The full raw JSON data.
    """

    def __init__(self, raw: dict) -> None:
        self.raw = raw
        self.molecule: Molecule | None = None
        self.calc_params: CalculationParameters | None = None
        self.response_params: ResponseParameters | None = None
        self.energy: float | None = None
        self.alpha: dict | None = None
        self.beta: dict | None = None
        self.raman: dict | None = None

    def __repr__(self) -> str:
        parts = [f"energy={self.energy}"]
        if self.molecule:
            parts.append(f"molecule={self.molecule}")
        if self.alpha:
            parts.append("alpha=<present>")
        if self.beta:
            parts.append("beta=<present>")
        if self.raman:
            parts.append("raman=<present>")
        return f"CalculationResult({', '.join(parts)})"


def load_output(path: str | Path) -> CalculationResult:
    """Load a MADNESS JSON output file.

    Detects the format (MADQC calc_info or legacy output/outputs) and
    extracts molecule, parameters, energy, alpha, and beta data.

    Parameters
    ----------
    path : str or Path
        Path to a JSON output file.

    Returns
    -------
    CalculationResult
        Parsed output data.

    Examples
    --------
    >>> result = load_output("calc_info.json")
    >>> print(result.energy)
    >>> print(result.molecule)
    """
    path = Path(path)
    with open(path) as f:
        data = json.load(f)

    result = CalculationResult(raw=data)

    if "tasks" in data:
        _parse_calc_info(data, result)
    elif "return_energy" in data:
        _parse_moldft_output(data, result)
    else:
        _parse_legacy_output(data, result)

    return result


# ---------------------------------------------------------------------------
# MADQC calc_info.json format
# ---------------------------------------------------------------------------


def _parse_calc_info(data: dict, result: CalculationResult) -> None:
    """Parse MADQC ``*.calc_info.json`` format."""
    tasks = data.get("tasks", [])
    if not tasks:
        return

    # Use the first (or most complete) task
    task = tasks[0]

    # Molecule
    mol_data = task.get("molecule")
    if mol_data:
        result.molecule = _build_molecule_from_json(mol_data)

    # Energy — look in multiple places
    opt = task.get("optimization_results", {})
    if "final_energy" in opt:
        result.energy = opt["final_energy"]
    elif "energy" in task:
        result.energy = task["energy"]
    elif "energy" in task.get("properties", {}):
        result.energy = task["properties"]["energy"]

    # Calculation parameters from convergence info
    conv = task.get("convergence", {})
    params: dict[str, Any] = {}
    if "converged_for_dconv" in conv:
        params["dconv"] = conv["converged_for_dconv"]
    if "converged_for_thresh" in conv:
        params["econv"] = conv["converged_for_thresh"]

    # Look for calculation_parameters in task
    calc_data = task.get("calculation_parameters", {})
    for key in ("xc", "k", "maxiter", "protocol", "localize", "econv", "dconv"):
        if key in calc_data:
            params[key] = calc_data[key]

    if params:
        result.calc_params = CalculationParameters(**params)

    # Alpha / beta from response tasks; raman from properties
    for t in tasks:
        resp = t.get("response", {})
        if "alpha" in resp:
            result.alpha = resp["alpha"]
        if "beta" in resp:
            result.beta = resp["beta"]
        props = t.get("properties", {})
        if "raman_spectra" in props:
            result.raman = props["raman_spectra"]


# ---------------------------------------------------------------------------
# Direct moldft calc_info.json format (return_energy at top level)
# ---------------------------------------------------------------------------


def _parse_moldft_output(data: dict, result: CalculationResult) -> None:
    """Parse direct moldft ``*.calc_info.json`` format.

    This format is produced by running ``moldft`` directly (not via MADQC).
    Energy is stored at ``data["return_energy"]`` and molecule geometry is
    nested under ``data["molecule"]``.
    """
    result.energy = data.get("return_energy")

    mol_data = data.get("molecule")
    if mol_data:
        result.molecule = _build_molecule_from_json(mol_data)

    calc_data = data.get("parameters", {})
    params: dict[str, Any] = {}
    for key in ("xc", "k", "maxiter", "localize", "econv", "dconv"):
        if key in calc_data:
            params[key] = calc_data[key]
    if params:
        result.calc_params = CalculationParameters(**params)


# ---------------------------------------------------------------------------
# Legacy output.json / outputs.json format
# ---------------------------------------------------------------------------


def _parse_legacy_output(data: dict, result: CalculationResult) -> None:
    """Parse legacy ``output.json`` or ``outputs.json`` format."""
    # Energy
    moldft = data.get("moldft", {})
    if "energy" in moldft:
        result.energy = moldft["energy"]

    # Alpha
    resp = data.get("response", {})
    if "alpha" in resp:
        result.alpha = resp["alpha"]

    # Beta
    hyper = data.get("hyper", {})
    if "beta" in hyper:
        result.beta = hyper["beta"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_molecule_from_json(mol_data: dict) -> Molecule:
    """Build a Molecule from JSON molecule data."""
    symbols = mol_data.get("symbols", [])
    geometry = mol_data.get("geometry", [])
    params = mol_data.get("parameters", {})
    units = params.get("units", "atomic")

    atoms_parts = []
    for sym, coord in zip(symbols, geometry):
        atoms_parts.append(f"{sym} {coord[0]:.12f} {coord[1]:.12f} {coord[2]:.12f}")
    atoms_str = "; ".join(atoms_parts)

    kwargs: dict[str, Any] = {"units": units}
    if "eprec" in params:
        kwargs["eprec"] = params["eprec"]
    if params.get("no_orient"):
        kwargs["no_orient"] = True
    if "core_type" in params and params["core_type"] != "none":
        kwargs["core_type"] = params["core_type"]

    return Molecule(atoms=atoms_str, **kwargs)
