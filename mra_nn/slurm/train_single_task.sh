#!/bin/bash
#SBATCH --partition=a100-long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --job-name=mra-nn-st
#SBATCH --output=/gpfs/projects/rjh/ruhin/mra_nn/logs/train_st_%j.out
#SBATCH --error=/gpfs/projects/rjh/ruhin/mra_nn/logs/train_st_%j.err

export PATH="/cm/shared/apps/slurm/21.08.8/bin:$PATH"
source /gpfs/projects/rjh/ruhin/mra_nn/.venv/bin/activate
export PYTHONPATH=/gpfs/projects/rjh/ruhin/madness-ruhin:/gpfs/projects/rjh/adrian/pymra/src:${PYTHONPATH:-}

cd /gpfs/projects/rjh/ruhin/madness-ruhin

python mra_nn/train.py \
    --config mra_nn/configs/single_task.yaml
