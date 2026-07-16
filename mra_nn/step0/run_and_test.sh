#!/bin/bash
#SBATCH --job-name=mra-nn-step0-run
#SBATCH --partition=long-40core
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=0:30:00
#SBATCH --output=/gpfs/projects/rjh/ruhin/mra_nn/step0/logs/run.out
#SBATCH --error=/gpfs/projects/rjh/ruhin/mra_nn/step0/logs/run.err

source /etc/profile.d/modules.sh
module load shared gcc/13.2.0 cmake/3.27.0
INTEL_PATH="/gpfs/software/intel/oneAPI/2024_2"
source "${INTEL_PATH}/setvars.sh" --force

set -euo pipefail

STEP0_DIR=/gpfs/projects/rjh/ruhin/mra_nn/step0
BUILD_DIR=/gpfs/projects/rjh/ruhin/madness-build-hdf5
MOLDFT=$BUILD_DIR/src/apps/moldft/moldft
DUMP=$BUILD_DIR/src/apps/molresponse/dump_mra_trees

echo "=== Step 0 (run-only): moldft + dump_mra_trees --coeffs ==="
echo "moldft        : $MOLDFT"
echo "dump_mra_trees: $DUMP"

# ── 1. Run moldft on H2O ──────────────────────────────────────────────────────
MOLDFT_DIR=$STEP0_DIR/moldft_h2o
rm -rf "$MOLDFT_DIR" && mkdir -p "$MOLDFT_DIR"
cp "$STEP0_DIR/h2o.in" "$MOLDFT_DIR/input"
cd "$MOLDFT_DIR"

export MAD_NUM_THREADS=39
echo "[1/2] Running moldft (H2O, thresh=1e-4, k=6)..."
mpirun -np 1 "$MOLDFT" 2>&1 | tee "$STEP0_DIR/logs/moldft.log"
echo "[1/2] moldft done."
ls -lh mad.restartdata* || { echo "ERROR: no restartdata produced"; exit 1; }

# ── 2. Smoke test: dump_mra_trees --coeffs ────────────────────────────────────
SMOKE_OUT=$STEP0_DIR/smoke_out
rm -rf "$SMOKE_OUT" && mkdir -p "$SMOKE_OUT"

echo "[2/2] Running dump_mra_trees --coeffs..."
mpirun -np 1 "$DUMP" \
  --archive="$MOLDFT_DIR/mad.restartdata" \
  --out="$SMOKE_OUT" \
  --coeffs \
  2>&1 | tee "$STEP0_DIR/logs/dump.log"

echo ""
echo "=== Output files ==="
ls -lh "$SMOKE_OUT"/

H5_COUNT=$(ls "$SMOKE_OUT"/*.mad.h5 2>/dev/null | wc -l)
if [ "$H5_COUNT" -gt 0 ]; then
  echo "GATE PASSED: $H5_COUNT .mad.h5 file(s) produced."
  for f in "$SMOKE_OUT"/*.mad.h5; do
    echo "  $f:"
    python3 -c "import h5py,sys; f=h5py.File(sys.argv[1],'r'); print('    keys:', list(f.keys())); [print('   ',k,dict(f[k].attrs)) for k in f.keys()]" "$f" || true
  done
else
  echo "GATE FAILED: no .mad.h5 files in $SMOKE_OUT"
  exit 1
fi

echo ""
echo "=== Step 0 COMPLETE ==="
