"""Integration test — Madness.compute() end-to-end submit path.

Runs a real moldft calculation on the dev cluster.  Skipped automatically
when the binary is absent.

Env vars (cluster defaults used when not set):
  MADNESS_EXECUTABLE        — path to moldft binary
  MADNESS_MPI_COMMAND       — mpi launcher + num-procs flag (e.g. "mpiexec -n")
  MADNESS_LD_LIBRARY_PATH   — extra colon-separated LD_LIBRARY_PATH entries
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Cluster-specific defaults (SeaWulf / xeonmax nodes)
# ---------------------------------------------------------------------------

_MOLDFT   = "/gpfs/projects/rjh/ruhin/madness-build/src/apps/moldft/moldft"
_MPIEXEC  = "/gpfs/software/openmpi/xeonmax/gcc13.2/4.1.6/bin/mpiexec"
_EXTRA_LD = (
    "/gpfs/software/gcc/13.2.0/lib64"
    ":/gpfs/software/intel/oneAPI/2024_2/tbb/2021.13/lib"
)

_BINARY   = os.environ.get("MADNESS_EXECUTABLE", _MOLDFT)
_MPI_CMD  = os.environ.get("MADNESS_MPI_COMMAND", f"{_MPIEXEC} -n")
_EXTRA_LD = os.environ.get("MADNESS_LD_LIBRARY_PATH", _EXTRA_LD)

requires_madness = pytest.mark.skipif(
    not Path(_BINARY).exists(),
    reason=f"MADNESS binary not found at {_BINARY!r}. Set MADNESS_EXECUTABLE.",
)


def _patch_ld(monkeypatch: pytest.MonkeyPatch) -> None:
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    new_val = f"{_EXTRA_LD}:{existing}" if existing else _EXTRA_LD
    monkeypatch.setenv("LD_LIBRARY_PATH", new_val)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@requires_madness
class TestMadnessCompute:
    """End-to-end: write → submit → read using Madness.compute()."""

    def test_h2_ground_state_energy(self, tmp_path, monkeypatch):
        """Compute H2 LDA/MRA energy and verify it matches the known value."""
        from madnessproject import CalculationParameters, Molecule
        from madnessproject.program import ComputeSettings, Madness

        _patch_ld(monkeypatch)

        mol = Molecule(
            atoms="H 0.0 0.0 -0.37; H 0.0 0.0 0.37",
            units="atomic",
        )
        calc = CalculationParameters(xc="lda", maxiter=20)

        settings = ComputeSettings(
            work_dir=str(tmp_path),
            mpi_command=_MPI_CMD,
            mpi_num_procs=1,
            MAD_NUM_THREADS=1,
            madness_executable=_BINARY,
        )

        result = Madness.compute(mol, calc, compute_settings=settings)

        assert result.energy is not None, "energy must not be None after a real run"
        assert result.energy == pytest.approx(-0.9150068, rel=1e-4), (
            f"H2 LDA energy {result.energy} deviates from expected -0.9150068"
        )

    def test_h2_dry_run_writes_input(self, tmp_path, monkeypatch):
        """dry_run=True writes the input file but does not call moldft."""
        from madnessproject import CalculationParameters, Molecule
        from madnessproject.program import ComputeSettings, Madness

        _patch_ld(monkeypatch)

        mol = Molecule(atoms="H 0.0 0.0 -0.37; H 0.0 0.0 0.37", units="atomic")
        calc = CalculationParameters(xc="lda")

        settings = ComputeSettings(
            work_dir=str(tmp_path),
            mpi_command=_MPI_CMD,
            mpi_num_procs=1,
            madness_executable=_BINARY,
        )

        result = Madness.compute(mol, calc, compute_settings=settings, dry_run=True)

        assert (tmp_path / "mad.in").exists(), "Input file must be written in dry_run"
        assert result.energy is None, "dry_run result should have no energy"
        assert not any(tmp_path.glob("*.calc_info.json")), (
            "moldft must not have been called in dry_run"
        )
