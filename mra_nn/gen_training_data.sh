#!/bin/bash
# Generate MRA-NN training data (rho0, vnuc, rho) for 15 molecules.
#
# Step A (no SCF): dump_training_functions --input=mol.in
#   → rho0.mad.h5, vnuc.mad.h5
#
# Step B (SCF convergence): moldft --input=mol.in
#   → mad.restartdata
#
# Step C (converged rho): dump_training_functions --input=mol.in --archive=mad.restartdata
#   → rho.mad.h5
#
# Run this script from the mra_nn/ directory after building MADNESS with HDF5:
#   cd /gpfs/projects/rjh/ruhin/mra_nn
#   sbatch gen_training_data.sh

#SBATCH --job-name=mra_nn_data
#SBATCH --partition=long-40core
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --time=12:00:00
#SBATCH --output=/gpfs/projects/rjh/ruhin/mra_nn/logs/gen_training_data_%j.out
#SBATCH --error=/gpfs/projects/rjh/ruhin/mra_nn/logs/gen_training_data_%j.err

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
MOLS_DIR=/gpfs/projects/rjh/ruhin/mra_nn/molecules
DATA_DIR=/gpfs/projects/rjh/ruhin/mra_nn/training_data
MOLECULES=(h2o nh3 ch4 co2 hf n2 co hcn c2h2 c2h4 c2h6 h2co ch3oh h2o2 hcl)

export MAD_NUM_THREADS=$((SLURM_CPUS_PER_TASK - 1))

mkdir -p logs "$DATA_DIR"

for mol in "${MOLECULES[@]}"; do
    echo "==============================="
    echo "Molecule: $mol"
    echo "==============================="

    MOL_IN="$MOLS_DIR/${mol}.in"
    MOL_DIR="$DATA_DIR/${mol}"
    mkdir -p "$MOL_DIR"

    # ---- Step A: rho0 + vnuc (no SCF) ----------------------------------------
    echo "[${mol}] Step A: generating rho0 + vnuc (no SCF) ..."
    cd "$MOL_DIR"
    mpirun -np 1 "$DUMP" \
        --input="$MOL_IN" \
        --out="$MOL_DIR"
    cd -

    # ---- Step B: run moldft to convergence ------------------------------------
    echo "[${mol}] Step B: running moldft ..."
    cd "$MOL_DIR"
    cp "$MOL_IN" input
    mpirun -np 1 "$MOLDFT"
    cd -

    # ---- Step C: converged rho ------------------------------------------------
    # Find the restart archive produced by moldft.
    # ParallelInputArchive appends ".00000" itself, so pass the prefix only.
    ARCHIVE_FILE=$(ls "$MOL_DIR"/mad.restartdata.00000 2>/dev/null | head -1 || \
                   ls "$MOL_DIR"/moldft.restartdata.00000 2>/dev/null | head -1 || true)
    ARCHIVE="${ARCHIVE_FILE%.00000}"
    if [ -n "$ARCHIVE" ]; then
        echo "[${mol}] Step C: extracting converged rho from $ARCHIVE ..."
        cd "$MOL_DIR"
        mpirun -np 1 "$DUMP" \
            --input="$MOL_IN" \
            --archive="$ARCHIVE" \
            --out="$MOL_DIR"
        cd -
    else
        echo "[${mol}] WARNING: no restart archive found in $MOL_DIR; skipping rho export"
    fi

    echo "[${mol}] Done. Contents of $MOL_DIR:"
    ls -lh "$MOL_DIR"/*.mad.h5 2>/dev/null || echo "  (no .mad.h5 files yet)"
    echo ""
done

echo "All molecules processed. Training data in: $DATA_DIR"
