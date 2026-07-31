#!/bin/bash
# Generate MRA-NN training data (rho0, vnuc, rho) for ch3f only.
# Same pipeline as gen_training_data.sh but for a single molecule.
#
#SBATCH --job-name=mra_ch3f
#SBATCH --partition=long-40core
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --time=4:00:00
#SBATCH --output=/gpfs/projects/rjh/ruhin/mra_nn/logs/gen_ch3f_%j.out
#SBATCH --error=/gpfs/projects/rjh/ruhin/mra_nn/logs/gen_ch3f_%j.err

source /etc/profile.d/modules.sh
module load shared gcc/13.2.0 cmake/3.27.0

INTEL_PATH="/gpfs/software/intel/oneAPI/2024_2"
# Source setvars.sh before set -u — it references unbound vars internally
source "${INTEL_PATH}/setvars.sh" --force
export TBB_DIR=${INTEL_PATH}/tbb/2021.13/lib/cmake/tbb
export LD_LIBRARY_PATH=${INTEL_PATH}/tbb/2021.13/lib/intel64/gcc4.8:${LD_LIBRARY_PATH:-}

set -euo pipefail

export PATH=/cm/shared/apps/slurm/21.08.8/bin:$PATH

BUILD=/gpfs/projects/rjh/ruhin/madness-build-hdf5
MOLDFT=$BUILD/src/apps/moldft/moldft
DUMP=$BUILD/src/apps/molresponse/dump_training_functions
MOL=ch3f
MOL_IN=/gpfs/projects/rjh/ruhin/mra_nn/molecules/${MOL}.in
MOL_DIR=/gpfs/projects/rjh/ruhin/mra_nn/training_data/${MOL}

export MAD_NUM_THREADS=$((SLURM_CPUS_PER_TASK - 1))

mkdir -p "$MOL_DIR"

echo "==============================="
echo "Molecule: $MOL"
echo "==============================="

# ---- Step A: rho0 + vnuc (no SCF) --------------------------------------------
echo "[${MOL}] Step A: generating rho0 + vnuc ..."
cd "$MOL_DIR"
mpirun -np 1 "$DUMP" \
    --input="$MOL_IN" \
    --out="$MOL_DIR"
cd -

# ---- Step B: run moldft to convergence ---------------------------------------
echo "[${MOL}] Step B: running moldft ..."
cd "$MOL_DIR"
cp "$MOL_IN" input
mpirun -np 1 "$MOLDFT"
cd -

# ---- Step C: converged rho ---------------------------------------------------
ARCHIVE_FILE=$(ls "$MOL_DIR"/mad.restartdata.00000 2>/dev/null | head -1 || \
               ls "$MOL_DIR"/moldft.restartdata.00000 2>/dev/null | head -1 || true)
ARCHIVE="${ARCHIVE_FILE%.00000}"
if [ -n "$ARCHIVE" ]; then
    echo "[${MOL}] Step C: extracting converged rho from $ARCHIVE ..."
    cd "$MOL_DIR"
    mpirun -np 1 "$DUMP" \
        --input="$MOL_IN" \
        --archive="$ARCHIVE" \
        --out="$MOL_DIR"
    cd -
else
    echo "[${MOL}] ERROR: no restart archive found — moldft may have failed"
    exit 1
fi

echo "[${MOL}] Done."
ls -lh "$MOL_DIR"/*.mad.h5
