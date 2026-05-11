"""CalculationParameterss class for MADNESS calculations.

Maps to the ``dft`` section of a MADNESS ``.in`` file.
Named CalculationParameterss (not DFT) to reflect that it holds all
ground-state calculation parameters, not just DFT-specific ones.

Mirrors the daltonproject QCMethod pattern: construct an object with
keyword arguments, then pass it to ``madness_input()`` to render.
"""

from __future__ import annotations



class Protocol:
    """Multi-level convergence protocol for MADNESS.

    A protocol is a list of convergence thresholds applied in sequence,
    from coarse to fine.

    Examples
    --------
    >>> p = Protocol([1e-4, 1e-6])
    >>> p = Protocol.default()
    >>> p = Protocol.tight()
    """

    def __init__(self, thresholds: list[float]) -> None:
        if not thresholds:
            raise ValueError("Protocol must have at least one threshold.")
        self.thresholds = sorted(thresholds, reverse=True)

    @classmethod
    def default(cls) -> Protocol:
        return cls([1e-4, 1e-6])

    @classmethod
    def tight(cls) -> Protocol:
        return cls([1e-4, 1e-6, 1e-7])

    @classmethod
    def very_tight(cls) -> Protocol:
        return cls([1e-4, 1e-6, 1e-7, 1e-8])

    def __repr__(self) -> str:
        return f"Protocol({self.thresholds})"

    def to_list(self) -> list[float]:
        return list(self.thresholds)


class CalculationParameters:
    """Ground-state calculation parameters for MADNESS.

    Maps to the ``dft`` block in a MADNESS ``.in`` file.  Mirrors
    ``daltonproject.QCMethod`` — construct with keyword arguments,
    then pass to :func:`madness_input` to generate the input string.

    Parameters
    ----------
    xc : str
        Exchange-correlation functional: ``"hf"``, ``"lda"``, ``"b3lyp"``,
        ``"pbe0"``, etc.  Default ``"hf"``.
    k : int, optional
        Wavelet order (6-10 typical; -1 = auto from thresh).
    l : float, optional
        User coordinate box size in atomic units.
    charge : float, optional
        Total molecular charge (overrides Molecule.charge if set).
    econv : float, optional
        Energy convergence threshold.
    dconv : float, optional
        Density convergence threshold.
    maxiter : int, optional
        Maximum SCF iterations.
    protocol : list[float] or Protocol, optional
        Multi-level convergence protocol.
    restart : bool, optional
        Restart from orbitals on disk.
    save : bool, optional
        Save orbitals to disk.
    localize : str, optional
        Localization method: ``"pm"``, ``"boys"``, ``"new"``, ``"canon"``.
    pointgroup : str, optional
        Point group: ``"c1"``, ``"c2v"``, ``"d2h"``, etc.
    spin_restricted : bool, optional
        Spin-restricted calculation.
    nopen : int, optional
        Number of unpaired electrons.
    dipole : bool, optional
        Calculate dipole moment at each SCF step.
    derivatives : bool, optional
        Calculate nuclear derivatives.
    gopt : bool, optional
        Geometry optimization.
    gtol : float, optional
        Geometry optimization gradient tolerance.
    gmaxiter : int, optional
        Geometry optimization max iterations.
    algopt : str, optional
        Optimization algorithm: ``"bfgs"``, ``"cg"``.
    print_level : int, optional
        Verbosity: 0=none, 1=final, 2=iter, 3=timings, 10=debug.
    maxsub : int, optional
        DIIS subspace size (0/1 = disable).
    aobasis : str, optional
        AO basis for initial guess.
    lo : float, optional
        Smallest length scale to resolve.
    maxrotn : float, optional
        Max orbital rotation per SCF step.

    Examples
    --------
    >>> calc = CalculationParameters(xc="hf", econv=1e-7, maxiter=20)
    >>> calc = CalculationParameters(
    ...     dipole=True, econv=1e-7, maxiter=20,
    ...     protocol=[1e-4, 1e-6], gopt=True,
    ... )
    """

    def __init__(
        self,
        xc: str = "hf",
        k: int | None = None,
        l: float | None = None,
        charge: float | None = None,
        econv: float | None = None,
        dconv: float | None = None,
        maxiter: int | None = None,
        protocol: list[float] | Protocol | None = None,
        restart: bool | None = None,
        save: bool | None = None,
        localize: str | None = None,
        pointgroup: str | None = None,
        spin_restricted: bool | None = None,
        nopen: int | None = None,
        dipole: bool | None = None,
        derivatives: bool | None = None,
        gopt: bool | None = None,
        gtol: float | None = None,
        gmaxiter: int | None = None,
        algopt: str | None = None,
        print_level: int | None = None,
        maxsub: int | None = None,
        aobasis: str | None = None,
        lo: float | None = None,
        maxrotn: float | None = None,
    ) -> None:
        self.xc = xc
        self.k = k
        self.l = l
        self.charge = charge
        self.econv = econv
        self.dconv = dconv
        self.maxiter = maxiter
        if isinstance(protocol, Protocol):
            self.protocol = protocol.to_list()
        else:
            self.protocol = protocol
        self.restart = restart
        self.save = save
        self.localize = localize
        self.pointgroup = pointgroup
        self.spin_restricted = spin_restricted
        self.nopen = nopen
        self.dipole = dipole
        self.derivatives = derivatives
        self.gopt = gopt
        self.gtol = gtol
        self.gmaxiter = gmaxiter
        self.algopt = algopt
        self.print_level = print_level
        self.maxsub = maxsub
        self.aobasis = aobasis
        self.lo = lo
        self.maxrotn = maxrotn

    @property
    def method(self) -> str:
        """Return the method name (uppercased xc or 'HF')."""
        return self.xc.upper()

    @property
    def settings(self) -> dict:
        """Return dft-section parameters as a dict (None values omitted).

        ``xc`` is only included when it is not ``"hf"`` (MADNESS default).
        """
        d: dict = {}
        if self.xc.lower() != "hf":
            d["xc"] = self.xc
        for attr in (
            "dipole", "k", "l", "charge", "econv", "dconv", "maxiter",
            "protocol", "gopt", "gtol", "gmaxiter", "algopt",
            "derivatives", "restart", "save", "localize", "pointgroup",
            "spin_restricted", "nopen", "print_level", "maxsub", "aobasis",
            "lo", "maxrotn",
        ):
            val = getattr(self, attr)
            if val is not None:
                d[attr] = val
        return d

    def __repr__(self) -> str:
        pairs = ", ".join(f"{k}={v!r}" for k, v in self.settings.items())
        return f"CalculationParameters(xc={self.xc!r}, {pairs})" if pairs else f"CalculationParameters(xc={self.xc!r})"
