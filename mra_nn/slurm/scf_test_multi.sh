#!/bin/bash
# Approach 3: Multi-molecule SCF test with level-clamped model (levels 10-14)
#
# Tests whether fine-level density improvements (0.41-0.98x at levels 10-14)
# translate to fewer SCF iterations across multiple molecules and thresholds.
#
# Molecules: val (ethanol, so2, hnnn) + test (h2o2, c2h2, glyoxal) + ch3oh (reference)
# Thresholds: 1e-6 (standard) and 1e-8 (tighter — deeper trees, more fine-level nodes)
#
#SBATCH --partition=long-40core
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=6:00:00
#SBATCH --mem=40G
#SBATCH --job-name=scf-multi
#SBATCH --output=/gpfs/projects/rjh/ruhin/mra_nn/logs/scf_multi_%j.out
#SBATCH --error=/gpfs/projects/rjh/ruhin/mra_nn/logs/scf_multi_%j.err

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

CHECKPOINT=$DATA/checkpoints/2026-08-10_06-02/best.pt
MODEL_LEVELS="10,11,12,13,14"
RESULTS=$DATA/scf_test_multi
mkdir -p "$RESULTS"

# Molecule name -> electron count
declare -A ELECTRONS=(
    [ch3oh]=18
    [ethanol]=26
    [so2]=32
    [hnnn]=22
    [h2o2]=18
    [c2h2]=14
    [glyoxal]=30
)

MOLECULES=(ch3oh ethanol so2 hnnn h2o2 c2h2 glyoxal)
THRESHOLDS=(1e-6 1e-8)

# ============================================================================
# Step 0: Build binaries on this node (same as scf_test.sh)
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
echo "  h5_to_archive OK"
echo ""

# ============================================================================
# Run tests for each molecule x threshold combination
# ============================================================================

SUMMARY_FILE="$RESULTS/summary.txt"
echo "Multi-Molecule SCF Test — $(date)" > "$SUMMARY_FILE"
echo "Checkpoint: $CHECKPOINT" >> "$SUMMARY_FILE"
echo "Model levels: $MODEL_LEVELS" >> "$SUMMARY_FILE"
echo "=========================================" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

for MOL in "${MOLECULES[@]}"; do
    NELEC=${ELECTRONS[$MOL]}

    for THRESH in "${THRESHOLDS[@]}"; do
        TAG="${MOL}_${THRESH}"
        TESTDIR="$RESULTS/$TAG"
        mkdir -p "$TESTDIR"

        echo ""
        echo "###################################################################"
        echo "### $MOL @ thresh=$THRESH  (${NELEC} electrons)"
        echo "###################################################################"

        # --- Step 1: Generate ML-predicted density ---
        echo "  [1/5] Predicting density with ML model..."
        cd "$SRC"

        $PYTHON mra_nn/predict.py \
            --checkpoint "$CHECKPOINT" \
            --rho0 "$DATA/training_data/${MOL}/rho0.mad.h5" \
            --vnuc "$DATA/training_data/${MOL}/vnuc.mad.h5" \
            --n-electrons "$NELEC" \
            --use-model-levels "$MODEL_LEVELS" \
            --out "$TESTDIR/rhoML.mad.h5"

        echo "    rhoML.mad.h5 written"

        # --- Step 2: Convert HDF5 to MADNESS binary archive ---
        echo "  [2/5] Converting HDF5 to MADNESS archive..."
        cd "$TESTDIR"
        mpirun -np 1 "$H5_TO_ARCHIVE" rhoML.mad.h5 rhoML || true

        if [ ! -f rhoML.00000 ]; then
            echo "  ERROR: rhoML.00000 not written for $MOL — skipping"
            echo "$TAG: FAILED (archive conversion)" >> "$SUMMARY_FILE"
            continue
        fi

        # --- Step 3: Create molecule input with this threshold ---
        # Replace thresh in the molecule input file
        sed "s/thresh 1e-6/thresh $THRESH/" "$DATA/molecules/${MOL}.in" > "$TESTDIR/input_template"

        # --- Step 4: Baseline SCF ---
        echo "  [3/5] Baseline SCF (rho0 initial guess)..."
        BASELINE_DIR="$TESTDIR/baseline"
        mkdir -p "$BASELINE_DIR"
        cp "$TESTDIR/input_template" "$BASELINE_DIR/input"
        cd "$BASELINE_DIR"
        rm -f *.restartdata* rhoML.00000

        mpirun -np 1 "$MOLDFT" 2>&1 | tee moldft_baseline.log

        # --- Step 5: ML-guess SCF ---
        echo "  [4/5] ML-guess SCF (rhoML initial guess)..."
        ML_DIR="$TESTDIR/ml_guess"
        mkdir -p "$ML_DIR"
        cp "$TESTDIR/input_template" "$ML_DIR/input"
        cp "$TESTDIR/rhoML.00000" "$ML_DIR/rhoML.00000"
        cd "$ML_DIR"
        rm -f *.restartdata*

        mpirun -np 1 "$MOLDFT" 2>&1 | tee moldft_ml.log

        # --- Step 6: Extract results ---
        echo "  [5/5] Extracting results..."

        BL_ENERGY=$(grep "final energy" "$BASELINE_DIR/moldft_baseline.log" | tail -1 || echo "N/A")
        ML_ENERGY=$(grep "final energy" "$ML_DIR/moldft_ml.log" | tail -1 || echo "N/A")

        # Count total iterations (lines matching "iteration")
        BL_ITERS=$(grep -c "iteration" "$BASELINE_DIR/moldft_baseline.log" 2>/dev/null || echo "0")
        ML_ITERS=$(grep -c "iteration" "$ML_DIR/moldft_ml.log" 2>/dev/null || echo "0")

        echo "  RESULT: $TAG"
        echo "    Baseline: $BL_ITERS iterations, $BL_ENERGY"
        echo "    ML-guess: $ML_ITERS iterations, $ML_ENERGY"

        # Write to summary
        echo "--- $TAG ---" >> "$SUMMARY_FILE"
        echo "  Baseline: $BL_ITERS iters | $BL_ENERGY" >> "$SUMMARY_FILE"
        echo "  ML-guess: $ML_ITERS iters | $ML_ENERGY" >> "$SUMMARY_FILE"
        echo "" >> "$SUMMARY_FILE"

        # Save per-protocol iteration detail
        echo "--- Baseline protocol detail ($TAG) ---" > "$TESTDIR/iteration_detail.txt"
        grep -E "proto|iteration|converged" "$BASELINE_DIR/moldft_baseline.log" >> "$TESTDIR/iteration_detail.txt" 2>/dev/null || true
        echo "" >> "$TESTDIR/iteration_detail.txt"
        echo "--- ML-guess protocol detail ($TAG) ---" >> "$TESTDIR/iteration_detail.txt"
        grep -E "proto|iteration|converged" "$ML_DIR/moldft_ml.log" >> "$TESTDIR/iteration_detail.txt" 2>/dev/null || true

    done
done

# ============================================================================
# Final summary
# ============================================================================
echo ""
echo "========================================="
echo "FINAL SUMMARY"
echo "========================================="
cat "$SUMMARY_FILE"
echo ""
echo "Detailed results in: $RESULTS/"
echo "========================================="
