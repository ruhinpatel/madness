"""Tests for madnessproject — input generation and output loading."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

from madnessproject import (
    CalculationParameters,
    Molecule,
    ResponseParameters,
    load_output,
    madness_input,
)

FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture()
def water_atomic() -> Molecule:
    """H2O in atomic units matching the raman fixture geometry."""
    return Molecule(
        atoms="O 0.0 0.0 0.21293780; H 0.0 1.42131521 -0.85175395; H 0.0 -1.42131521 -0.85175395",
        units="atomic",
        eprec=1e-6,
        no_orient=True,
    )


def _normalize_value(raw: str) -> str:
    """Normalize a MADNESS scalar or list value for canonical comparison.

    Converts floats to ``repr()`` form so that ``0.0001`` and ``1e-04``
    compare equal, and ``1e-7`` and ``1e-07`` compare equal.
    Booleans are lower-cased.  String tokens (e.g. ``xyz``) are kept as-is.
    """
    raw = raw.strip()
    # List [a,b,c,...]
    if raw.startswith("[") and raw.endswith("]"):
        inner = raw[1:-1]
        items = [_normalize_value(x.strip()) for x in inner.split(",") if x.strip()]
        return "[" + ",".join(items) + "]"
    # Boolean
    if raw.lower() in ("true", "false"):
        return raw.lower()
    # Try numeric
    try:
        return repr(float(raw))
    except ValueError:
        return raw


def _parse_block_kv(text: str, block_name: str) -> dict[str, str]:
    """Extract normalized key→value pairs from a named block in .in text.

    Lines inside the block are stripped of comments (``# ...``), whitespace
    is collapsed, and values are passed through :func:`_normalize_value`.
    The opening ``<block_name>`` line and closing ``end`` are not included.
    """
    kv: dict[str, str] = {}
    in_block = False
    for line in text.splitlines():
        stripped = line.split("#")[0].strip()
        if not stripped:
            continue
        if stripped == block_name:
            in_block = True
            continue
        if in_block:
            if stripped == "end":
                break
            parts = stripped.split(None, 1)
            if len(parts) == 2:
                key, raw_val = parts
                kv[key] = _normalize_value(raw_val)
    return kv


_ATOM_LINE = re.compile(
    r"^([A-Z][a-z]?)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)"
)


def _parse_atoms(text: str) -> list[tuple[str, float, float, float]]:
    """Extract ``(symbol, x, y, z)`` tuples from the molecule block of .in text."""
    atoms: list[tuple[str, float, float, float]] = []
    in_mol = False
    for line in text.splitlines():
        stripped = line.split("#")[0].strip()
        if stripped == "molecule":
            in_mol = True
            continue
        if in_mol:
            if stripped == "end":
                break
            m = _ATOM_LINE.match(stripped)
            if m:
                atoms.append((m.group(1), float(m.group(2)), float(m.group(3)), float(m.group(4))))
    return atoms


# ---------------------------------------------------------------------------
# Input generation — block order and content
# ---------------------------------------------------------------------------

class TestMadnessInput:
    def test_block_order_dft_molecule_response(self, water_atomic):
        """dft block must come before molecule, molecule before response."""
        calc = CalculationParameters(dipole=True, econv=1e-7, maxiter=20, gopt=True,
                                     protocol=[1e-4, 1e-6, 1e-7])
        resp = ResponseParameters(
            dipole=True,
            frequencies=[0.0, 0.02, 0.04, 0.06, 0.08, 0.1],
            quadratic=True,
            nuclear=True,
            nuclear_directions="xyz",
            nuclear_frequencies=0.0,
            nuclear_atom_indices=[0, 1, 2],
        )
        content = madness_input(calc, water_atomic, resp)
        dft_pos = content.index("dft\n")
        mol_pos = content.index("molecule\n")
        resp_pos = content.index("response\n")
        assert dft_pos < mol_pos < resp_pos, (
            f"Expected dft < molecule < response, got positions {dft_pos}, {mol_pos}, {resp_pos}"
        )

    def test_dft_block_contains_expected_keys(self, water_atomic):
        calc = CalculationParameters(dipole=True, econv=1e-7, maxiter=20, gopt=True)
        content = madness_input(calc, water_atomic)
        assert "dipole true" in content
        assert "econv 1e-07" in content
        assert "maxiter 20" in content
        assert "gopt true" in content

    def test_molecule_block_contains_atoms(self, water_atomic):
        calc = CalculationParameters()
        content = madness_input(calc, water_atomic)
        assert "O " in content
        assert "H " in content
        assert "units atomic" in content

    def test_response_block_contains_frequencies(self, water_atomic):
        calc = CalculationParameters()
        resp = ResponseParameters(frequencies=[0.0, 0.02, 0.04])
        content = madness_input(calc, water_atomic, resp)
        assert "dipole.frequencies" in content
        assert "0.0,0.02,0.04" in content

    def test_no_response_omits_response_block(self, water_atomic):
        calc = CalculationParameters()
        content = madness_input(calc, water_atomic)
        assert "response" not in content

    def test_emit_property_flag(self, water_atomic):
        calc = CalculationParameters()
        resp = ResponseParameters(emit_property=True)
        content = madness_input(calc, water_atomic, resp)
        assert "property true" in content

    def test_suppress_property_flag(self, water_atomic):
        calc = CalculationParameters()
        resp = ResponseParameters(emit_property=False)
        content = madness_input(calc, water_atomic, resp)
        # "property true" should not appear; "property" key absent
        assert "property true" not in content


# ---------------------------------------------------------------------------
# Golden-file test — generated input vs. fixture
# ---------------------------------------------------------------------------

class TestGoldenFile:
    """Compare madness_input() output against tests/fixtures/load_calc/03_mra-raman_h2o/mad.h2o_gopt.in.

    The fixture was generated by MADNESS with its own formatting conventions
    (variable indentation, comments, abbreviated exponents).  We compare
    semantically: DFT and response blocks are compared as key→value dicts
    (normalising float notation so ``0.0001`` == ``1e-04``); atom coordinates
    are checked with floating-point tolerance.
    """

    FIXTURE_IN = FIXTURES / "03_mra-raman_h2o" / "mad.h2o_gopt.in"

    def _make_params(self) -> tuple[CalculationParameters, ResponseParameters]:
        calc = CalculationParameters(
            dipole=True,
            econv=1e-7,
            maxiter=20,
            gopt=True,
            protocol=[1e-4, 1e-6, 1e-7],
        )
        resp = ResponseParameters(
            dipole=True,
            frequencies=[0.0, 0.02, 0.04, 0.06, 0.08, 0.1],
            quadratic=True,
            nuclear=True,
            nuclear_directions="xyz",
            nuclear_frequencies=0.0,
            nuclear_atom_indices=[0, 1, 2],
            requested_properties=["polarizability", "raman"],
        )
        return calc, resp

    def test_dft_block_matches_fixture(self, water_atomic):
        """DFT block key/value pairs must match the fixture (float-normalised)."""
        calc, resp = self._make_params()
        generated = madness_input(calc, water_atomic, resp)

        gen_kv = _parse_block_kv(generated, "dft")
        fix_kv = _parse_block_kv(self.FIXTURE_IN.read_text(), "dft")

        assert gen_kv == fix_kv, (
            f"DFT block mismatch:\n  generated: {gen_kv}\n  fixture:   {fix_kv}"
        )

    def test_response_block_matches_fixture(self, water_atomic):
        """Response block key/value pairs must match the fixture (float-normalised)."""
        calc, resp = self._make_params()
        generated = madness_input(calc, water_atomic, resp)

        gen_kv = _parse_block_kv(generated, "response")
        fix_kv = _parse_block_kv(self.FIXTURE_IN.read_text(), "response")

        assert gen_kv == fix_kv, (
            f"Response block mismatch:\n  generated: {gen_kv}\n  fixture:   {fix_kv}"
        )

    def test_atom_geometry_matches_fixture(self, water_atomic):
        """Atom symbols and coordinates must match the fixture geometry."""
        calc, resp = self._make_params()
        generated = madness_input(calc, water_atomic, resp)

        gen_atoms = _parse_atoms(generated)
        fix_atoms = _parse_atoms(self.FIXTURE_IN.read_text())

        assert len(gen_atoms) == len(fix_atoms), (
            f"Atom count mismatch: {len(gen_atoms)} vs {len(fix_atoms)}"
        )
        for i, (ga, fa) in enumerate(zip(gen_atoms, fix_atoms)):
            assert ga[0] == fa[0], f"Atom {i} symbol mismatch: {ga[0]} vs {fa[0]}"
            assert ga[1] == pytest.approx(fa[1], abs=1e-6), f"Atom {i} x mismatch"
            assert ga[2] == pytest.approx(fa[2], abs=1e-6), f"Atom {i} y mismatch"
            assert ga[3] == pytest.approx(fa[3], abs=1e-6), f"Atom {i} z mismatch"


# ---------------------------------------------------------------------------
# Output loading — round-trip against real fixtures
# ---------------------------------------------------------------------------

class TestLoadOutput:
    def test_n2_energy_not_none(self):
        """load_output on n2 output.json must return a non-None energy."""
        path = FIXTURES / "01_mra-d04_n2" / "output.json"
        result = load_output(path)
        assert result.energy is not None, "energy should not be None for n2 fixture"

    def test_n2_energy_value(self):
        """n2 SCF energy matches fixture value."""
        path = FIXTURES / "01_mra-d04_n2" / "output.json"
        result = load_output(path)
        assert result.energy == pytest.approx(-413.6193973843323, rel=1e-6)

    def test_bh2cl_beta_not_none(self):
        """load_output on bh2cl outputs.json must return non-None beta."""
        path = FIXTURES / "00_mra-d06_bh2cl" / "outputs.json"
        result = load_output(path)
        assert result.beta is not None, "beta should not be None for bh2cl fixture"

    def test_bh2cl_alpha_shape(self):
        """BH2Cl alpha must have n_freqs*9 entries (9 tensor components per freq)."""
        path = FIXTURES / "00_mra-d06_bh2cl" / "outputs.json"
        result = load_output(path)
        assert result.alpha is not None
        n_entries = len(result.alpha["alpha"])
        assert n_entries % 9 == 0, f"Expected multiple of 9 alpha entries, got {n_entries}"
        n_freqs = n_entries // 9
        # BH2Cl fixture has 9 frequency points
        assert n_freqs == 9, f"Expected 9 frequency points, got {n_freqs}"
        # Confirm ij labels cycle through 9 components
        assert len(set(result.alpha["ij"][:9])) == 9

    def test_bh2cl_alpha_spot_check(self):
        """BH2Cl alpha[0] (XX at omega=0) must match fixture value."""
        path = FIXTURES / "00_mra-d06_bh2cl" / "outputs.json"
        result = load_output(path)
        assert result.alpha is not None
        assert result.alpha["omega"][0] == pytest.approx(0.0, abs=1e-9)
        assert result.alpha["ij"][0] == "XX"
        assert result.alpha["alpha"][0] == pytest.approx(23.39689143776393, rel=1e-6)

    def test_bh2cl_beta_spot_check(self):
        """BH2Cl Beta[2] (XXZ component at first frequency) must match fixture value."""
        path = FIXTURES / "00_mra-d06_bh2cl" / "outputs.json"
        result = load_output(path)
        assert result.beta is not None
        # Index 2 is the XXZ component (A='X', B='X', C='Z')
        assert result.beta["A"][2] == "X"
        assert result.beta["B"][2] == "X"
        assert result.beta["C"][2] == "Z"
        assert result.beta["Beta"][2] == pytest.approx(-1.900607932290064, rel=1e-6)

    def test_calc_info_loads_without_error(self):
        """load_output on calc_info.json format must not raise."""
        path = FIXTURES / "03_mra-raman_h2o" / "mad.h2o_gopt.calc_info.json"
        result = load_output(path)
        assert result is not None

    def test_raman_h2o_raman_not_none(self):
        """load_output on raman H2O calc_info must return non-None raman."""
        path = FIXTURES / "03_mra-raman_h2o" / "mad.h2o_gopt.calc_info.json"
        result = load_output(path)
        assert result.raman is not None, "raman should not be None for H2O raman fixture"

    def test_raman_h2o_raman_has_expected_keys(self):
        """Raman dict must contain the standard MADQC output keys."""
        path = FIXTURES / "03_mra-raman_h2o" / "mad.h2o_gopt.calc_info.json"
        result = load_output(path)
        assert result.raman is not None
        expected_keys = {
            "polarizability_derivatives",
            "polarizability_derivatives_normal_modes",
            "polarization_frequencies",
            "raman_spectra",
            "vibrational_frequencies",
        }
        assert expected_keys <= set(result.raman.keys()), (
            f"Missing raman keys: {expected_keys - set(result.raman.keys())}"
        )
