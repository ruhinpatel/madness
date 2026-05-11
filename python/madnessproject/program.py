"""ComputeSettings and Madness program runner.

Mirrors ``daltonproject.program.ComputeSettings`` and
``daltonproject.dalton.Dalton.compute()`` for the MADNESS code.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from .calculation_parameters import CalculationParameters
from .input_generator import write_input
from .molecule import Molecule
from .output_reader import CalculationResult, load_output
from .response_parameters import ResponseParameters


class ComputeSettings:
    """Compute settings for a MADNESS calculation.

    Mirrors ``daltonproject.ComputeSettings`` with MADNESS-specific fields.

    Parameters
    ----------
    work_dir : str, optional
        Directory where input/output files are written. Defaults to CWD.
    scratch_dir : str, optional
        Scratch directory for temporary storage. Defaults to ``$SCRATCH``
        or the system temp directory.
    mpi_num_procs : int, optional
        Number of MPI processes. Defaults to ``$SLURM_NTASKS`` if set,
        otherwise 1.
    MAD_NUM_THREADS : int, optional
        Number of threads per MPI rank (sets ``MAD_NUM_THREADS``
        environment variable). Defaults to ``$MAD_NUM_THREADS`` or 1.
    omp_num_threads : int, optional
        Alias for compatibility with daltonproject. Sets OMP_NUM_THREADS.
    mpi_command : str, optional
        MPI launcher command (e.g. ``"mpirun -np"``). Defaults to
        ``$MADNESS_MPI_COMMAND`` or ``"mpirun -np"``.
    mpi_num_procs : int, optional
        Number of MPI processes.
    memory : int, optional
        Memory per node in MB.
    madness_executable : str, optional
        Path to the MADNESS executable (e.g. ``moldft`` or ``molresponse``).
        Defaults to looking up ``$MADNESS_EXECUTABLE`` or ``moldft`` on PATH.

    Examples
    --------
    >>> settings = ComputeSettings(
    ...     work_dir="/gpfs/scratch/user/h2o_calc",
    ...     MAD_NUM_THREADS=5,
    ...     mpi_num_procs=8,
    ... )
    """

    def __init__(
        self,
        work_dir: str | None = None,
        scratch_dir: str | None = None,
        mpi_num_procs: int | None = None,
        MAD_NUM_THREADS: int | None = None,
        omp_num_threads: int | None = None,
        mpi_command: str | None = None,
        memory: int | None = None,
        madness_executable: str | None = None,
    ) -> None:
        self.work_dir: str = work_dir if work_dir is not None else os.getcwd()

        if scratch_dir is None:
            scratch_dir = os.environ.get("SCRATCH") or os.environ.get("TMPDIR")
        self.scratch_dir: str | None = scratch_dir

        if mpi_num_procs is None:
            slurm_ntasks = os.environ.get("SLURM_NTASKS")
            mpi_num_procs = int(slurm_ntasks) if slurm_ntasks else 1
        self.mpi_num_procs: int = mpi_num_procs

        if MAD_NUM_THREADS is None:
            MAD_NUM_THREADS = int(os.environ.get("MAD_NUM_THREADS", "1"))
        self.MAD_NUM_THREADS: int = MAD_NUM_THREADS

        if omp_num_threads is None:
            omp_num_threads = int(os.environ.get("OMP_NUM_THREADS", "1"))
        self.omp_num_threads: int = omp_num_threads

        if mpi_command is None:
            mpi_command = os.environ.get("MADNESS_MPI_COMMAND", "mpirun -np")
        self.mpi_command: str = mpi_command

        self.memory: int | None = memory

        if madness_executable is None:
            madness_executable = os.environ.get("MADNESS_EXECUTABLE", "moldft")
        self.madness_executable: str = madness_executable

    def env(self) -> dict[str, str]:
        """Return environment variables to set when running MADNESS."""
        env = os.environ.copy()
        env["MAD_NUM_THREADS"] = str(self.MAD_NUM_THREADS)
        env["OMP_NUM_THREADS"] = str(self.omp_num_threads)
        if self.scratch_dir:
            env["MAD_SCRATCH"] = self.scratch_dir
        return env

    def launch_command(self, input_file: Path) -> list[str]:
        """Return the command to launch MADNESS."""
        parts = self.mpi_command.split() + [str(self.mpi_num_procs)]
        parts += [self.madness_executable, str(input_file)]
        return parts

    def __repr__(self) -> str:
        return (
            f"ComputeSettings(work_dir={self.work_dir!r}, "
            f"mpi_num_procs={self.mpi_num_procs}, "
            f"MAD_NUM_THREADS={self.MAD_NUM_THREADS})"
        )


class Madness:
    """MADNESS program driver. Mirrors ``daltonproject.dalton.Dalton``."""

    @staticmethod
    def compute(
        molecule: Molecule,
        calculation_parameters: CalculationParameters,
        response_parameters: ResponseParameters | None = None,
        compute_settings: ComputeSettings | None = None,
        *,
        input_filename: str = "mad.in",
        output_filename: str = "mad.out",
        dry_run: bool = False,
    ) -> CalculationResult:
        """Run a MADNESS calculation and return parsed results.

        Mirrors ``Dalton.compute(molecule, basis, method, props, compute_settings=...)``
        but without a ``basis`` argument (MADNESS has no AO basis).

        Parameters
        ----------
        molecule : Molecule
            Molecular geometry.
        calculation_parameters : CalculationParameters
            Ground-state calculation parameters.
        response_parameters : ResponseParameters, optional
            Response parameters. If None, only ground-state is computed.
        compute_settings : ComputeSettings, optional
            Runtime settings. Defaults are inferred from environment.
        input_filename : str
            Input file name relative to ``work_dir``.
        output_filename : str
            Output file name (stdout log) relative to ``work_dir``.
        dry_run : bool
            If True, write input files but do not execute MADNESS.

        Returns
        -------
        CalculationResult
            Parsed results. When ``dry_run=True`` the result contains
            the molecule/parameters but no energy/alpha/beta data.
        """
        if compute_settings is None:
            compute_settings = ComputeSettings()

        work_dir = Path(compute_settings.work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)

        input_path = work_dir / input_filename
        write_input(
            calculation_parameters,
            molecule,
            response_parameters,
            output_path=input_path,
        )

        if dry_run:
            result = CalculationResult(raw={})
            result.molecule = molecule
            result.calc_params = calculation_parameters
            result.response_params = response_parameters
            return result

        if shutil.which(compute_settings.madness_executable.split()[0]) is None \
                and not Path(compute_settings.madness_executable).exists():
            raise FileNotFoundError(
                f"MADNESS executable not found: {compute_settings.madness_executable}. "
                f"Set MADNESS_EXECUTABLE or pass madness_executable=... to ComputeSettings."
            )

        output_path = work_dir / output_filename
        cmd = compute_settings.launch_command(input_path)
        with open(output_path, "w") as f_out:
            subprocess.run(
                cmd,
                cwd=work_dir,
                env=compute_settings.env(),
                stdout=f_out,
                stderr=subprocess.STDOUT,
                check=True,
            )

        # Locate and parse a JSON output file
        for candidate in (
            work_dir / "calc_info.json",
            work_dir / "output.json",
            work_dir / "outputs.json",
        ):
            if candidate.exists():
                return load_output(candidate)

        # Fallback: also try *.calc_info.json pattern
        for p in work_dir.glob("*.calc_info.json"):
            return load_output(p)

        raise FileNotFoundError(
            f"No MADNESS JSON output found in {work_dir}. "
            f"Expected calc_info.json, output.json, or outputs.json."
        )
