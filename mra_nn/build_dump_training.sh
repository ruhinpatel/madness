#!/bin/bash
# Build dump_training_functions on a compute node (needs Intel OneAPI for MPI).
# Submit with: sbatch build_dump_training.sh

#SBATCH --job-name=build-dump-training
#SBATCH --partition=long-40core
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --time=01:00:00
#SBATCH --output=/gpfs/projects/rjh/ruhin/mra_nn/logs/build_dump_training_%j.out
#SBATCH --error=/gpfs/projects/rjh/ruhin/mra_nn/logs/build_dump_training_%j.err

source /etc/profile.d/modules.sh
module load shared gcc/13.2.0 cmake/3.27.0

INTEL_PATH="/gpfs/software/intel/oneAPI/2024_2"
# Source setvars.sh before set -u — it references unbound vars internally
source "${INTEL_PATH}/setvars.sh" --force
export TBB_DIR=${INTEL_PATH}/tbb/2021.13/lib/cmake/tbb
export LD_LIBRARY_PATH=${INTEL_PATH}/tbb/2021.13/lib/intel64/gcc4.8:${LD_LIBRARY_PATH:-}

set -euo pipefail

BUILD_DIR=/gpfs/projects/rjh/ruhin/madness-build-hdf5
SRC_DIR=/gpfs/projects/rjh/ruhin/madness-ruhin
NPROC=${SLURM_CPUS_PER_TASK:-96}

mkdir -p /gpfs/projects/rjh/ruhin/mra_nn/logs

echo "Building dump_training_functions ..."
echo "Source : $SRC_DIR  (branch: $(git -C $SRC_DIR rev-parse --abbrev-ref HEAD))"
echo "Build  : $BUILD_DIR"

cd "$BUILD_DIR"

# Full reconfigure — 40-core nodes have matching libraries for MPI/HDF5.
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
  -DBUILD_TESTING=OFF

cmake --build . --target dump_training_functions -- -j$NPROC

BIN="$BUILD_DIR/src/apps/molresponse/dump_training_functions"
echo "Build complete: $BIN"
ls -lh "$BIN"
