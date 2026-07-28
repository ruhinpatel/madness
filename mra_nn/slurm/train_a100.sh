#!/bin/bash
#SBATCH --partition=a100-long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --job-name=mra-nn-train
#SBATCH --output=/gpfs/projects/rjh/ruhin/mra_nn/logs/train_%j.out
#SBATCH --error=/gpfs/projects/rjh/ruhin/mra_nn/logs/train_%j.err

set -euo pipefail

export PATH="/cm/shared/apps/slurm/21.08.8/bin:$PATH"
source /gpfs/projects/rjh/ruhin/mra_nn/.venv/bin/activate
export PYTHONPATH=/gpfs/projects/rjh/adrian/pymra/src:${PYTHONPATH:-}

CONFIG="${1:-/gpfs/projects/rjh/ruhin/madness-ruhin/mra_nn/configs/default.yaml}"

echo "=== MRA-NN Training ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Config: $CONFIG"
echo ""

python /gpfs/projects/rjh/ruhin/madness-ruhin/mra_nn/train.py --config "$CONFIG"
