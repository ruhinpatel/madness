"""ResponseParameters class for MADNESS response calculations.

Maps to the ``response`` section of a MADNESS ``.in`` file.
Mirrors ``daltonproject.Property`` — specify which properties to compute
and their parameters, then pass to ``madness_input()`` to render.
"""

from __future__ import annotations


class ResponseParameters:
    """Response calculation parameters for MADNESS.

    Maps to the ``response`` block in a MADNESS ``.in`` file.  Mirrors
    ``daltonproject.Property`` — construct with keyword arguments,
    then pass to :func:`madness_input` to generate the input string.

    Parameters
    ----------
    dipole : bool
        Compute dipole (polarizability) response.  Default True.
    frequencies : list[float], optional
        Optical frequencies in Hartree for dipole response.
    directions : str
        Perturbation directions: ``"xyz"``, ``"x"``, ``"y"``, ``"z"``.
    emit_property : bool
        Emit the ``property true`` flag.  Default True.
    requested_properties : list[str], optional
        Properties to output.  Default ``["polarizability"]`` when
        ``dipole=True`` and no other perturbation set.
    quadratic : bool, optional
        Compute quadratic (hyperpolarizability) response.
    nuclear : bool, optional
        Compute nuclear (Raman) response.
    nuclear_directions : str, optional
        Directions for nuclear response.
    nuclear_frequencies : float, optional
        Frequency for nuclear response.
    nuclear_atom_indices : list[int], optional
        Atom indices for nuclear response.
    maxiter : int, optional
        Maximum response iterations.
    dconv : float, optional
        Density convergence threshold.
    protocol : list[float], optional
        Multi-level convergence protocol.
    kain : bool, optional
        Use Krylov Accelerated Inexact Newton solver.
    maxsub : int, optional
        KAIN subspace size.
    maxrotn : float, optional
        Max orbital rotation per response step.
    save : bool, optional
        Save response orbitals to disk.
    restart : bool, optional
        Restart from saved response orbitals.
    localize : str, optional
        Localization: ``"pm"``, ``"boys"``, ``"new"``, ``"canon"``.
    print_level : int, optional
        Verbosity level.
    prefix : str, optional
        Prefix for output files.
    archive : str, optional
        Path to ground-state restart data.

    Examples
    --------
    >>> resp = ResponseParameters(dipole=True, frequencies=[0.0, 0.0656])
    >>> resp = ResponseParameters(quadratic=True, frequencies=[0.0])
    >>> resp = ResponseParameters(nuclear=True, nuclear_atom_indices=[0, 1, 2])
    """

    def __init__(
        self,
        dipole: bool = True,
        frequencies: list[float] | None = None,
        directions: str = "xyz",
        emit_property: bool = True,
        requested_properties: list[str] | None = None,
        quadratic: bool | None = None,
        nuclear: bool | None = None,
        nuclear_directions: str | None = None,
        nuclear_frequencies: float | None = None,
        nuclear_atom_indices: list[int] | None = None,
        maxiter: int | None = None,
        dconv: float | None = None,
        protocol: list[float] | None = None,
        kain: bool | None = None,
        maxsub: int | None = None,
        maxrotn: float | None = None,
        save: bool | None = None,
        restart: bool | None = None,
        localize: str | None = None,
        print_level: int | None = None,
        prefix: str | None = None,
        archive: str | None = None,
    ) -> None:
        self.dipole = dipole
        self.frequencies = frequencies if frequencies is not None else [0.0]
        self.directions = directions
        self.emit_property = emit_property
        self.requested_properties = requested_properties
        self.quadratic = quadratic
        self.nuclear = nuclear
        self.nuclear_directions = nuclear_directions
        self.nuclear_frequencies = nuclear_frequencies
        self.nuclear_atom_indices = nuclear_atom_indices
        self.maxiter = maxiter
        self.dconv = dconv
        self.protocol = protocol
        self.kain = kain
        self.maxsub = maxsub
        self.maxrotn = maxrotn
        self.save = save
        self.restart = restart
        self.localize = localize
        self.print_level = print_level
        self.prefix = prefix
        self.archive = archive

    @property
    def property_type(self) -> str:
        """Infer the property type from settings."""
        if self.nuclear:
            return "raman"
        if self.quadratic:
            return "beta"
        return "alpha"

    def _default_requested_properties(self) -> list[str]:
        """Infer requested_properties when user didn't supply any."""
        if self.nuclear:
            return ["polarizability", "raman"]
        if self.quadratic:
            return ["hyperpolarizability"]
        return ["polarizability"]

    @property
    def settings(self) -> dict:
        """Return response-section parameters as a dict.

        Uses MADNESS dot-notation keys (e.g. ``dipole.frequencies``).
        Block order matches Adrian's example format:
        dipole.frequencies, dipole.directions, requested_properties, property.
        """
        d: dict = {}

        # Dipole perturbation
        if self.dipole:
            d["dipole.frequencies"] = self.frequencies
            d["dipole.directions"] = self.directions

        # Quadratic
        if self.quadratic:
            d["quadratic"] = True

        # Nuclear (Raman)
        if self.nuclear:
            d["nuclear"] = True
            if self.nuclear_directions is not None:
                d["nuclear.directions"] = self.nuclear_directions
            if self.nuclear_frequencies is not None:
                d["nuclear.frequencies"] = self.nuclear_frequencies
            if self.nuclear_atom_indices is not None:
                d["nuclear.atom_indices"] = self.nuclear_atom_indices

        # requested_properties — default based on perturbation type
        if self.requested_properties is not None:
            d["requested_properties"] = self.requested_properties
        else:
            d["requested_properties"] = self._default_requested_properties()

        # property true/false flag
        if self.emit_property:
            d["property"] = True

        # Solver parameters
        for attr, key in (
            ("maxiter", "maxiter"),
            ("dconv", "dconv"),
            ("protocol", "protocol"),
            ("kain", "kain"),
            ("maxsub", "maxsub"),
            ("maxrotn", "maxrotn"),
            ("save", "save"),
            ("restart", "restart"),
            ("localize", "localize"),
            ("print_level", "print_level"),
            ("prefix", "prefix"),
            ("archive", "archive"),
        ):
            val = getattr(self, attr)
            if val is not None:
                d[key] = val

        return d

    def __repr__(self) -> str:
        return f"ResponseParameters(property={self.property_type!r}, frequencies={self.frequencies})"
