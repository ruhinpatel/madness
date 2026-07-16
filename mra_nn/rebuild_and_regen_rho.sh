#!/bin/bash
# Rebuild dump_training_functions (incremental) and regenerate only rho.mad.h5
# for all 5 molecules. rho0 and vnuc are unaffected by the factor-of-2 fix.
#
# sbatch rebuild_and_regen_rho.sh

#SBATCH --job-name=fix-rho-2x
#SBATCH --partition=long-40core
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/projects/rjh/ruhin/mra_nn/logs/rebuild_regen_rho_%j.out
#SBATCH --error=/gpfs/projects/rjh/ruhin/mra_nn/logs/rebuild_regen_rho_%j.err

source /etc/profile.d/modules.sh
module load shared gcc/13.2.0 cmake/3.27.0

INTEL_PATH="/gpfs/software/intel/oneAPI/2024_2"
source "${INTEL_PATH}/setvars.sh" --force
export TBB_DIR=${INTEL_PATH}/tbb/2021.13/lib/cmake/tbb
export LD_LIBRARY_PATH=${INTEL_PATH}/tbb/2021.13/lib/intel64/gcc4.8:${LD_LIBRARY_PATH:-}

set -euo pipefail

BUILD=/gpfs/projects/rjh/ruhin/madness-build-hdf5
DUMP=$BUILD/src/apps/molresponse/dump_training_functions
MOLS_DIR=/gpfs/projects/rjh/ruhin/mra_nn/molecules
DATA_DIR=/gpfs/projects/rjh/ruhin/mra_nn/training_data
MOLECULES=(h2o nh3 ch4 co2 hf)
NPROC=${SLURM_CPUS_PER_TASK:-40}

export PATH=/cm/shared/apps/slurm/21.08.8/bin:$PATH
export MAD_NUM_THREADS=$((SLURM_CPUS_PER_TASK - 1))

mkdir -p /gpfs/projects/rjh/ruhin/mra_nn/logs

# ---- Step 1: Incremental rebuild ----
echo "=== Rebuilding dump_training_functions (incremental) ==="
cd "$BUILD"
cmake --build . --target dump_training_functions -- -j$NPROC
echo "Build complete: $(ls -lh "$DUMP")"

# ---- Step 2: Regenerate rho.mad.h5 only (Step C) ----
for mol in "${MOLECULES[@]}"; do
    echo ""
    echo "=== Regenerating rho.mad.h5 for $mol ==="

    MOL_IN="$MOLS_DIR/${mol}.in"
    MOL_DIR="$DATA_DIR/${mol}"

    ARCHIVE_FILE=$(ls "$MOL_DIR"/mad.restartdata.00000 2>/dev/null | head -1 || true)
    ARCHIVE="${ARCHIVE_FILE%.00000}"

    if [ -z "$ARCHIVE" ]; then
        echo "[${mol}] WARNING: no restart archive found — skipping"
        continue
    fi

    echo "[${mol}] Archive: $ARCHIVE"
    echo "[${mol}] Old rho.mad.h5:"
    ls -lh "$MOL_DIR/rho.mad.h5" 2>/dev/null || echo "  (not found)"

    cd "$MOL_DIR"
    mpirun -np 1 "$DUMP" \
        --input="$MOL_IN" \
        --archive="$ARCHIVE" \
        --out="$MOL_DIR"
    cd -

    echo "[${mol}] New rho.mad.h5:"
    ls -lh "$MOL_DIR/rho.mad.h5"
done

echo ""
echo "=== Done. Regenerated rho.mad.h5 for all molecules ==="
