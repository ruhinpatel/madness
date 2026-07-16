#!/bin/bash
#SBATCH --job-name=mra-nn-step0
#SBATCH --partition=long-40core
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=4:00:00
#SBATCH --output=/gpfs/projects/rjh/ruhin/mra_nn/step0/logs/step0.out
#SBATCH --error=/gpfs/projects/rjh/ruhin/mra_nn/step0/logs/step0.err

source /etc/profile.d/modules.sh
module load shared gcc/13.2.0 cmake/3.27.0

INTEL_PATH="/gpfs/software/intel/oneAPI/2024_2"
# Disable set -e around setvars.sh — it uses internal commands that may return non-zero
source "${INTEL_PATH}/setvars.sh" --force
export TBB_DIR=${INTEL_PATH}/tbb/2021.13/lib/cmake/tbb
export LD_LIBRARY_PATH=${INTEL_PATH}/tbb/2021.13/lib/intel64/gcc4.8:${LD_LIBRARY_PATH:-}

set -euo pipefail

BUILD_DIR=/gpfs/projects/rjh/ruhin/madness-build-hdf5
SRC_DIR=/gpfs/projects/rjh/ruhin/madness-ruhin
STEP0_DIR=/gpfs/projects/rjh/ruhin/mra_nn/step0
NPROC=40

echo "=== STEP 0: Build feat/mra-nn-data with HDF5 + molresponse ==="
echo "Source : $SRC_DIR  (branch: $(git -C $SRC_DIR rev-parse --abbrev-ref HEAD))"
echo "Build  : $BUILD_DIR"
echo "Cores  : $NPROC"
echo ""

# ── 1. Reconfigure ───────────────────────────────────────────────────────────
echo "[1/4] Configuring CMake..."
cd "$BUILD_DIR"
rm -f CMakeCache.txt
rm -rf CMakeFiles

cmake "$SRC_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/gpfs/software/gcc/13.2.0/bin/g++ \
  -DCMAKE_C_COMPILER=/gpfs/software/gcc/13.2.0/bin/gcc \
  -DCMAKE_CXX_FLAGS="-march=native" \
  -DENABLE_MPI=ON \
  -DMADNESS_TASK_BACKEND=TBB \
  -DTBB_DIR=/gpfs/software/intel/oneAPI/2024_2/tbb/2021.13/lib/cmake/tbb \
  -DENABLE_MKL=ON \
  -DMKL_ROOT=/gpfs/software/intel/oneAPI/2024_2/mkl/2024.2 \
  -DENABLE_HDF5=ON \
  -DMADNESS_ENABLE_HDF5=ON \
  -DHDF5_ROOT=/cm/shared/apps/hdf5/1.12.1 \
  -DBUILD_TESTING=OFF \
  2>&1 | tee "$STEP0_DIR/logs/cmake.log"

echo "[1/4] CMake configure done."

# ── 2. Build moldft + dump_mra_trees ─────────────────────────────────────────
echo "[2/4] Building moldft and dump_mra_trees (j=$NPROC)..."
cmake --build . --target moldft dump_mra_trees -- -j$NPROC \
  2>&1 | tee "$STEP0_DIR/logs/build.log"

echo "[2/4] Build done."
MOLDFT=$BUILD_DIR/src/apps/moldft/moldft
DUMP=$BUILD_DIR/src/apps/molresponse/dump_mra_trees
echo "  moldft        : $MOLDFT"
echo "  dump_mra_trees: $DUMP"

# ── 3. Run moldft on H2O ─────────────────────────────────────────────────────
echo "[3/4] Running moldft on H2O (thresh=1e-4, k=6)..."
MOLDFT_DIR=$STEP0_DIR/moldft_h2o
mkdir -p "$MOLDFT_DIR"
cp "$STEP0_DIR/h2o.in" "$MOLDFT_DIR/input"
cd "$MOLDFT_DIR"
export MAD_NUM_THREADS=39
mpirun -np 1 "$MOLDFT" 2>&1 | tee "$STEP0_DIR/logs/moldft.log"
echo "[3/4] moldft done."
ls -lh mad.restartdata* 2>/dev/null || { echo "ERROR: no restartdata produced"; exit 1; }

# ── 4. Smoke test: dump_mra_trees --coeffs ────────────────────────────────────
echo "[4/4] Running dump_mra_trees --coeffs (smoke test)..."
SMOKE_OUT=$STEP0_DIR/smoke_out
mkdir -p "$SMOKE_OUT"
mpirun -np 1 "$DUMP" \
  --archive="$MOLDFT_DIR/mad.restartdata" \
  --out="$SMOKE_OUT" \
  --coeffs \
  2>&1 | tee "$STEP0_DIR/logs/dump.log"

echo ""
echo "=== Output files ==="
ls -lh "$SMOKE_OUT"/
echo ""

# Verify at least one .mad.h5 was produced
H5_COUNT=$(ls "$SMOKE_OUT"/*.mad.h5 2>/dev/null | wc -l)
if [ "$H5_COUNT" -gt 0 ]; then
  echo "GATE PASSED: $H5_COUNT .mad.h5 file(s) produced."
  # Quick sanity: print top-level HDF5 groups
  for f in "$SMOKE_OUT"/*.mad.h5; do
    echo "  $f:"
    h5ls "$f" 2>/dev/null || python3 -c "import h5py,sys; f=h5py.File(sys.argv[1]); print(list(f.keys()))" "$f" || true
  done
else
  echo "GATE FAILED: no .mad.h5 files found in $SMOKE_OUT"
  exit 1
fi

echo ""
echo "=== Step 0 complete. feat/mra-nn-data branch builds and dumps .mad.h5. ==="
