#!/bin/bash
#SBATCH --partition=short-96core
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --job-name=mra-nn-install
#SBATCH --output=/gpfs/projects/rjh/ruhin/mra_nn/logs/install_%j.out
#SBATCH --error=/gpfs/projects/rjh/ruhin/mra_nn/logs/install_%j.err

set -euo pipefail

echo "=== MRA-NN pip install ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo ""

/gpfs/projects/rjh/ruhin/mra_nn/.venv/bin/python -m pip install torch pyyaml \
    --no-user \
    --cache-dir /gpfs/projects/rjh/ruhin/.pip_cache

echo ""
echo "=== Verifying ==="
/gpfs/projects/rjh/ruhin/mra_nn/.venv/bin/python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
/gpfs/projects/rjh/ruhin/mra_nn/.venv/bin/python -c "import yaml; print(f'pyyaml {yaml.__version__}')"
echo "Done."
