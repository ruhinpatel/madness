"""madnessproject — object-based API for MADNESS calculations.

Mirrors the ``daltonproject`` package pattern.

Dalton:

    import daltonproject as dp
    molecule = dp.Molecule(input_file='water.xyz')
    basis = dp.Basis(basis='cc-pVDZ')
    method = dp.QCMethod('HF')
    props = dp.Property(polarizabilities=True)
    settings = dp.ComputeSettings(work_dir='...', mpi_num_procs=90)
    result = dp.dalton.Dalton.compute(molecule, basis, method, props,
                                      compute_settings=settings)

MADNESS:

    from gecko import madnessproject as mad
    mol = mad.Molecule(input_file='water.xyz')
    calc = mad.CalculationParameters(dipole=True, econv=1e-7, maxiter=20,
                                     protocol=[1e-4, 1e-6], gopt=True)
    resp = mad.ResponseParameters()
    settings = mad.ComputeSettings(work_dir='...', MAD_NUM_THREADS=5,
                                   mpi_num_procs=8)
    result = mad.Madness.compute(mol, calc, resp, compute_settings=settings)

Object mapping to MADNESS .in file sections:

    CalculationParameters  ->  dft ... end
    ResponseParameters     ->  response ... end
    Molecule               ->  molecule ... end

Block order in generated inputs: ``dft → molecule → response``.
"""

from .calculation_parameters import CalculationParameters, Protocol
from .input_generator import madness_input, write_input
from .molecule import Molecule
from .output_reader import CalculationResult, load_output
from .program import ComputeSettings, Madness
from .response_parameters import ResponseParameters

__all__ = [
    # Parameter objects (one per .in block)
    "CalculationParameters",
    "Molecule",
    "ResponseParameters",
    # Convergence protocol helper
    "Protocol",
    # Runtime / driver
    "ComputeSettings",
    "Madness",
    # Input/output helpers
    "madness_input",
    "write_input",
    "load_output",
    "CalculationResult",
]

__version__ = "0.0.0"
