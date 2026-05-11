"""Generate MADNESS .in files from object-based parameters.

Mirrors ``daltonproject.dalton.program.dalton_input`` — takes parameter
objects and renders a complete MADNESS input string.

Block order matches existing fixtures:
``dft → molecule → response``.
"""

from __future__ import annotations

from pathlib import Path

from .calculation_parameters import CalculationParameters
from .molecule import Molecule
from .response_parameters import ResponseParameters


# ---------------------------------------------------------------------------
# Value rendering
# ---------------------------------------------------------------------------


def _render_value(v: object, *, bool_style: str = "lower") -> str:
    """Render a value to MADNESS .in string.

    ``bool_style`` controls boolean formatting:
        "lower" -> true/false (used in dft and response blocks)
        "title" -> True/False (used in molecule block per Adrian's example)
    """
    if isinstance(v, bool):
        if bool_style == "title":
            return "True" if v else "False"
        return "true" if v else "false"
    if isinstance(v, list):
        # Special formatting for fields with spaces after commas in mol block
        inner = ", ".join(_render_value(x, bool_style=bool_style) for x in v) \
            if bool_style == "title" \
            else ",".join(_render_value(x, bool_style=bool_style) for x in v)
        return "[" + inner + "]"
    if isinstance(v, float) and v != 0.0 and abs(v) < 1e-2:
        return f"{v:.0e}"
    return str(v)


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------


def _render_dft_section(calc: CalculationParameters) -> str:
    """Render the ``dft ... end`` block (lowercase booleans, 4-space indent)."""
    lines = ["dft"]
    for key, val in calc.settings.items():
        lines.append(f"    {key} {_render_value(val, bool_style='lower')}")
    lines.append("end")
    return "\n".join(lines)


def _render_response_section(resp: ResponseParameters) -> str:
    """Render the ``response ... end`` block (lowercase booleans, 2-space indent)."""
    lines = ["response"]
    for key, val in resp.settings.items():
        lines.append(f"  {key} {_render_value(val, bool_style='lower')}")
    lines.append("end")
    return "\n".join(lines)


def _render_molecule_section(mol: Molecule) -> str:
    """Render the ``molecule ... end`` block with embedded geometry.

    Matches Adrian's example: title-case booleans, 2-space indent,
    ``core_type none`` rendered without quotes.
    """
    lines = ["molecule"]
    for key, val in mol.settings.items():
        lines.append(f"  {key} {_render_value(val, bool_style='title')}")

    # Coordinates in the user-specified units
    coords = mol.coordinates
    for sym, coord in zip(mol.elements, coords):
        lines.append(f"  {sym} {coord[0]} {coord[1]} {coord[2]}")
    lines.append("end")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def madness_input(
    calc: CalculationParameters,
    mol: Molecule,
    resp: ResponseParameters | None = None,
) -> str:
    """Generate a MADNESS ``.in`` file as a string.

    Mirrors ``daltonproject.dalton.program.dalton_input()`` — takes
    parameter objects and returns the complete input file content.

    Block order is ``dft → molecule → response`` to match existing fixtures.

    Parameters
    ----------
    calc : CalculationParameters
        Ground-state calculation parameters (``dft`` block).
    mol : Molecule
        Molecular geometry (``molecule`` block).
    resp : ResponseParameters, optional
        Response calculation parameters (``response`` block).
        If None, only ``dft`` and ``molecule`` blocks are written.

    Returns
    -------
    str
        Complete MADNESS ``.in`` file content.
    """
    sections = [_render_dft_section(calc)]
    sections.append("")
    sections.append(_render_molecule_section(mol))
    if resp is not None:
        sections.append("")
        sections.append(_render_response_section(resp))

    return "\n".join(sections) + "\n"


def write_input(
    calc: CalculationParameters,
    mol: Molecule,
    resp: ResponseParameters | None = None,
    *,
    output_path: str | Path,
) -> Path:
    """Write a MADNESS ``.in`` file to disk."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(madness_input(calc, mol, resp))
    return path
