"""Molecule class for MADNESS calculations.

Mirrors the daltonproject Molecule API but generates MADNESS molecule blocks.
"""

from __future__ import annotations

import os
from typing import Sequence

import numpy as np
import qcelemental as qcel


class Molecule:
    """Molecular geometry for MADNESS calculations.

    Defaults match the ``example_h2o_frequency_calc.in`` reference format
    (Angstrom units, field=[0,0,0], pure_ae=True, symtol=-0.01, core_type=none).

    Examples
    --------
    >>> mol = Molecule(atoms="O 0.0 0.0 0.0; H 0.0 0.0 0.96; H 0.0 0.96 0.0")
    >>> mol = Molecule(input_file="water.xyz")
    >>> mol = Molecule.from_qcelemental(qcel_mol)
    """

    def __init__(
        self,
        atoms: str | None = None,
        input_file: str | None = None,
        charge: int = 0,
        units: str = "angstrom",
        eprec: float | None = 1e-6,
        no_orient: bool = True,
        field: list[float] | None = None,
        psp_calc: bool = False,
        pure_ae: bool = True,
        symtol: float | None = -0.01,
        core_type: str = "none",
    ) -> None:
        """Initialize Molecule.

        Parameters
        ----------
        atoms : str
            Atoms as ``"El x y z; El x y z; ..."`` or newline-separated.
        input_file : str
            Path to ``.xyz`` file.
        charge : int
            Total molecular charge.
        units : str
            Coordinate units: ``"angstrom"`` (default) or ``"atomic"`` (Bohr).
        eprec : float
            Smoothing parameter for the nuclear potential. Default 1e-6.
        no_orient : bool
            If True, do not reorient the molecule. Default True.
        field : list[float], optional
            External electric field ``[fx, fy, fz]`` in atomic units.
            Default ``[0.0, 0.0, 0.0]``.
        psp_calc : bool
            Pseudopotential calculation flag. Default False.
        pure_ae : bool
            Pure all-electron flag. Default True.
        symtol : float
            Symmetry detection threshold. Default -0.01.
        core_type : str
            Core potential type. Default ``"none"``.
        """
        if atoms is not None and input_file is not None:
            raise ValueError("Specify either atoms or input_file, not both.")
        if atoms is None and input_file is None:
            raise ValueError("Specify either atoms or input_file.")

        self.charge: int = charge
        self.units: str = units.lower()
        self.elements: list[str] = []
        self._coordinates: np.ndarray = np.empty((0, 3))

        # molecule-section parameters
        self.eprec: float | None = eprec
        self.no_orient: bool = no_orient
        self.field: list[float] = field if field is not None else [0.0, 0.0, 0.0]
        self.psp_calc: bool = psp_calc
        self.pure_ae: bool = pure_ae
        self.symtol: float | None = symtol
        self.core_type: str = core_type

        if input_file is not None:
            self._read_xyz(input_file)
        elif atoms is not None:
            self._parse_atoms(atoms)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_qcelemental(
        cls,
        qcel_mol: qcel.models.Molecule,
        units: str = "angstrom",
        **kwargs,
    ) -> Molecule:
        """Create from a QCElemental Molecule (qcel geometry is always Bohr)."""
        geom = np.asarray(qcel_mol.geometry).reshape(-1, 3)
        symbols = list(qcel_mol.symbols)
        charge = int(round(qcel_mol.molecular_charge))

        if units.lower() == "angstrom":
            bohr2ang = qcel.constants.conversion_factor("bohr", "angstrom")
            geom = geom * bohr2ang

        atoms_str = "; ".join(
            f"{sym} {x:.10f} {y:.10f} {z:.10f}"
            for sym, (x, y, z) in zip(symbols, geom)
        )
        return cls(atoms=atoms_str, charge=charge, units=units, **kwargs)

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _parse_atoms(self, atoms: str) -> None:
        if ";" in atoms:
            lines = [ln.strip() for ln in atoms.split(";") if ln.strip()]
        elif "\n" in atoms:
            lines = [ln.strip() for ln in atoms.split("\n") if ln.strip()]
        else:
            lines = [atoms.strip()]

        elements: list[str] = []
        coords: list[list[float]] = []
        for line in lines:
            parts = line.split()
            if len(parts) < 4:
                raise ValueError(f"Expected 'Element x y z', got: {line!r}")
            elements.append(parts[0].title())
            coords.append([float(parts[1]), float(parts[2]), float(parts[3])])

        self.elements = elements
        self._coordinates = np.array(coords)

    def _read_xyz(self, filename: str) -> None:
        if not os.path.isfile(filename):
            raise FileNotFoundError(f"{filename} not found.")
        with open(filename) as f:
            lines = f.read().strip().splitlines()

        n_atoms = int(lines[0])
        atom_lines = lines[2 : 2 + n_atoms]
        atoms_str = "; ".join(atom_lines)
        # XYZ files are always in Angstrom
        self.units = "angstrom"
        self._parse_atoms(atoms_str)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def coordinates(self) -> np.ndarray:
        """Atomic coordinates as (n_atoms, 3) array in self.units."""
        return self._coordinates

    @property
    def coordinates_bohr(self) -> np.ndarray:
        """Coordinates in Bohr regardless of input units."""
        if self.units == "angstrom":
            ang2bohr = qcel.constants.conversion_factor("angstrom", "bohr")
            return self._coordinates * ang2bohr
        return self._coordinates

    @property
    def num_atoms(self) -> int:
        return len(self.elements)

    def to_qcelemental(self) -> qcel.models.Molecule:
        """Convert to a QCElemental Molecule (always Bohr)."""
        return qcel.models.Molecule(
            symbols=self.elements,
            geometry=self.coordinates_bohr.flatten().tolist(),
            molecular_charge=self.charge,
        )

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"Molecule(num_atoms={self.num_atoms}, charge={self.charge}, "
            f"units={self.units!r}, elements={self.elements})"
        )

    # ------------------------------------------------------------------
    # Settings dict (for input generation)
    # ------------------------------------------------------------------

    @property
    def settings(self) -> dict:
        """Return molecule-section parameters in Adrian's reference order."""
        d: dict = {}
        if self.eprec is not None:
            d["eprec"] = self.eprec
        d["field"] = self.field
        d["no_orient"] = self.no_orient
        d["psp_calc"] = self.psp_calc
        d["pure_ae"] = self.pure_ae
        if self.symtol is not None:
            d["symtol"] = self.symtol
        d["core_type"] = self.core_type
        d["units"] = self.units
        return d
