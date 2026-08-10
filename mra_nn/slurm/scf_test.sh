#!/bin/bash
# SCF convergence test: compare baseline (rho0) vs ML-predicted density (rhoML)
# on ch3oh (validation molecule, not in training set).
#
# This script builds h5_to_archive on the same node to avoid -march=native
# instruction set mismatches, then runs the full test.
#
#SBATCH --partition=long-40core
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH --job-name=scf_test
#SBATCH --output=/gpfs/projects/rjh/ruhin/mra_nn/logs/scf_test_%j.out
#SBATCH --error=/gpfs/projects/rjh/ruhin/mra_nn/logs/scf_test_%j.err

source /etc/profile.d/modules.sh
module load shared gcc/13.2.0

INTEL_PATH="/gpfs/software/intel/oneAPI/2024_2"
source "${INTEL_PATH}/setvars.sh" --force
export LD_LIBRARY_PATH=${INTEL_PATH}/tbb/2021.13/lib/intel64/gcc4.8:${LD_LIBRARY_PATH:-}
export PATH=${INTEL_PATH}/mpi/2021.13/bin:/cm/shared/apps/slurm/21.08.8/bin:$PATH

set -euo pipefail

BUILD=/gpfs/projects/rjh/ruhin/madness-build-hdf5
SRC=/gpfs/projects/rjh/ruhin/madness-ruhin
DATA=/gpfs/projects/rjh/ruhin/mra_nn
CXX=/gpfs/software/gcc/13.2.0/bin/g++
HDF5_ROOT=/cm/shared/apps/hdf5/1.12.1
PYTHON=$DATA/.venv/bin/python

export PYTHONPATH=/gpfs/projects/rjh/ruhin/madness-ruhin:/gpfs/projects/rjh/adrian/pymra/src:${PYTHONPATH:-}
export MAD_NUM_THREADS=0

TESTDIR=$DATA/scf_test/ch3oh
mkdir -p "$TESTDIR"

# ============================================================================
# Step 0: Build h5_to_archive and relink moldft on THIS node
# ============================================================================
echo "========================================="
echo "Step 0: Building binaries on this node"
echo "========================================="
cd "$BUILD"

CXXFLAGS="-march=native -std=c++17 -O3 -DNDEBUG -fPIC"
DEFINES="-DMADNESS_LINALG_USE_LAPACKE \
  -DMADNESS_MPI_HEADER=\"${INTEL_PATH}/mpi/2021.13/include/mpi.h\" \
  -DMAD_ROOT_DIR=\"$SRC\" \
  -DMRA_CHEMDATA_DIR=\"$SRC/src/madness/chem\""
INCLUDES="-I$BUILD/src/madness/chem -I$SRC/src/madness/chem \
  -I$SRC/src -I$BUILD/src \
  -I$BUILD/src/madness/world -I$SRC/src/madness/world \
  -I${INTEL_PATH}/tbb/2021.13/include -I${INTEL_PATH}/mpi/2021.13/include \
  -I$BUILD/src/madness/misc -I$SRC/src/madness/misc \
  -I${INTEL_PATH}/mkl/2024.2/include \
  -I$BUILD/src/madness/tensor -I$SRC/src/madness/tensor \
  -I$BUILD/src/madness/external/tinyxml -I$SRC/src/madness/external/tinyxml \
  -I$BUILD/src/madness/external/muParser -I$SRC/src/madness/external/muParser \
  -I$BUILD/src/madness/mra -I$SRC/src/madness/mra \
  -I$SRC/src/apps -I$BUILD -I$SRC"
LINK_FLAGS="-Xlinker --enable-new-dtags \
  -Xlinker -rpath -Xlinker ${INTEL_PATH}/mpi/2021.13/lib \
  -Xlinker -rpath -Xlinker ${INTEL_PATH}/mpi/2021.13/lib \
  -Xlinker --enable-new-dtags"
LIBS="$BUILD/src/madness/chem/libMADchem.a \
  $BUILD/src/madness/mra/libMADmra.a \
  $BUILD/src/madness/tensor/libMADlinalg.a \
  $BUILD/src/madness/tensor/libMADtensor.a \
  $BUILD/src/madness/misc/libMADmisc.a \
  $BUILD/src/madness/world/libMADworld.a \
  ${INTEL_PATH}/tbb/2021.13/lib/intel64/gcc4.8/libtbb.so.12 \
  ${INTEL_PATH}/mpi/2021.13/lib/libmpicxx.so \
  ${INTEL_PATH}/mpi/2021.13/lib/libmpi.so"
LINK_TAIL="-lrt -lpthread -ldl -Wl,--start-group \
  ${INTEL_PATH}/mkl/2024.2/lib/libmkl_intel_lp64.a \
  ${INTEL_PATH}/mkl/2024.2/lib/libmkl_core.a \
  ${INTEL_PATH}/mkl/2024.2/lib/libmkl_sequential.a \
  -Wl,--end-group -lm -ldl \
  $BUILD/src/madness/external/tinyxml/libMADtinyxml.a \
  $BUILD/src/madness/external/muParser/libMADmuparser.a"

# Recompile SCF.cc
echo "  Compiling SCF.cc..."
$CXX $CXXFLAGS $DEFINES $INCLUDES \
  -c "$SRC/src/madness/chem/SCF.cc" \
  -o src/madness/chem/CMakeFiles/MADchem-obj.dir/SCF.cc.o

# Rebuild libMADchem.a
echo "  Rebuilding libMADchem.a..."
cd src/madness/chem
ar rcs libMADchem.a CMakeFiles/MADchem-obj.dir/*.o
cd "$BUILD"

# Relink moldft
echo "  Relinking moldft..."
$CXX $CXXFLAGS $LINK_FLAGS \
  src/apps/moldft/CMakeFiles/moldft.dir/moldft.cc.o \
  -o src/apps/moldft/moldft \
  $LIBS $LINK_TAIL

MOLDFT=$BUILD/src/apps/moldft/moldft
echo "  moldft OK: $(ls -la $MOLDFT | awk '{print $5}') bytes"

# Build h5_to_archive
echo "  Compiling h5_to_archive..."
H5_SRC="$SRC/src/apps/molresponse/tools/h5_to_archive.cpp"
H5_OUT="$BUILD/src/apps/molresponse/h5_to_archive"

$CXX $CXXFLAGS $DEFINES -DMADNESS_HAS_HDF5 \
  $INCLUDES \
  -I"$BUILD/src/apps/molresponse" -I"$SRC/src/apps/molresponse" \
  -I"$HDF5_ROOT/include" \
  -c "$H5_SRC" -o "${H5_OUT}.o"

$CXX $CXXFLAGS $LINK_FLAGS \
  "${H5_OUT}.o" -o "$H5_OUT" \
  -Wl,-rpath,"$HDF5_ROOT/lib" \
  $LIBS "$HDF5_ROOT/lib/libhdf5.so" -lz -ldl -lm $LINK_TAIL

H5_TO_ARCHIVE=$H5_OUT
echo "  h5_to_archive OK: $(ls -la $H5_TO_ARCHIVE | awk '{print $5}') bytes"
echo ""

# ============================================================================
# Step 1: Generate ML-predicted density
# ============================================================================
echo "========================================="
echo "Step 1: Predicting density with ML model"
echo "========================================="
cd "$SRC"

$PYTHON mra_nn/predict.py \
    --checkpoint "$DATA/checkpoints/2026-08-10_06-02/best.pt" \
    --rho0 "$DATA/training_data/ch3oh/rho0.mad.h5" \
    --vnuc "$DATA/training_data/ch3oh/vnuc.mad.h5" \
    --n-electrons 18 \
    --use-model-levels "10,11,12,13,14" \
    --out "$TESTDIR/rhoML.mad.h5"

echo "  rhoML.mad.h5 written: $(ls -la "$TESTDIR/rhoML.mad.h5" | awk '{print $5}') bytes"

# ============================================================================
# Step 2: Convert HDF5 to MADNESS binary archive
# ============================================================================
echo ""
echo "========================================="
echo "Step 2: Converting HDF5 to MADNESS archive"
echo "========================================="
cd "$TESTDIR"

# h5_to_archive may abort during MPI finalize — ignore exit code if output exists
mpirun -np 1 "$H5_TO_ARCHIVE" rhoML.mad.h5 rhoML || true

if [ ! -f rhoML.00000 ]; then
    echo "ERROR: rhoML.00000 not written"
    exit 1
fi
echo "  rhoML.00000 written: $(ls -la rhoML.00000 | awk '{print $5}') bytes"

# ============================================================================
# Step 3: Baseline SCF (normal rho0 initial guess)
# ============================================================================
echo ""
echo "========================================="
echo "Step 3: Baseline SCF (rho0 initial guess)"
echo "========================================="

BASELINE_DIR="$TESTDIR/baseline"
mkdir -p "$BASELINE_DIR"
cp "$DATA/molecules/ch3oh.in" "$BASELINE_DIR/input"
cd "$BASELINE_DIR"
rm -f *.restartdata* rhoML.00000

echo "  Running moldft (baseline)..."
mpirun -np 1 "$MOLDFT" 2>&1 | tee moldft_baseline.log

# ============================================================================
# Step 4: ML-guess SCF (rhoML initial guess)
# ============================================================================
echo ""
echo "========================================="
echo "Step 4: ML-guess SCF (rhoML initial guess)"
echo "========================================="

ML_DIR="$TESTDIR/ml_guess"
mkdir -p "$ML_DIR"
cp "$DATA/molecules/ch3oh.in" "$ML_DIR/input"
cp "$TESTDIR/rhoML.00000" "$ML_DIR/rhoML.00000"
cd "$ML_DIR"
rm -f *.restartdata*

echo "  Running moldft (ML guess)..."
mpirun -np 1 "$MOLDFT" 2>&1 | tee moldft_ml.log

# ============================================================================
# Step 5: Compare results
# ============================================================================
echo ""
echo "========================================="
echo "Step 5: Comparison"
echo "========================================="
echo ""
echo "=== BASELINE (rho0) ==="
grep "final energy" "$BASELINE_DIR/moldft_baseline.log" || echo "(not found)"
echo ""
echo "=== ML GUESS (rhoML) ==="
grep "final energy" "$ML_DIR/moldft_ml.log" || echo "(not found)"
echo ""

# Extract iteration counts from protocol lines
echo "--- Iteration details ---"
echo "Baseline iterations per protocol:"
grep -E "proto|iteration" "$BASELINE_DIR/moldft_baseline.log" | head -30
echo ""
echo "ML-guess iterations per protocol:"
grep -E "proto|iteration" "$ML_DIR/moldft_ml.log" | head -30

echo ""
echo "========================================="
echo "SCF test complete"
echo "========================================="
