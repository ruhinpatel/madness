#!/bin/bash
#SBATCH --job-name=rebuild-b
#SBATCH --partition=long-40core
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/gpfs/projects/rjh/ruhin/mra_nn/logs/rebuild_b_%j.out
#SBATCH --error=/gpfs/projects/rjh/ruhin/mra_nn/logs/rebuild_b_%j.err

set -euo pipefail

source /gpfs/projects/rjh/ruhin/mra_nn/.venv/bin/activate
export PYTHONPATH=/gpfs/projects/rjh/ruhin/madness-ruhin:/gpfs/projects/rjh/adrian/pymra/src:${PYTHONPATH:-}

DATA=/gpfs/projects/rjh/ruhin/mra_nn
OUT=$DATA/training_dataset.h5

# Back up existing dataset
if [ -f "$OUT" ]; then
    cp "$OUT" "${OUT}.bak-option-a"
    echo "Backed up existing dataset to ${OUT}.bak-option-a"
fi

python /gpfs/projects/rjh/ruhin/madness-ruhin/mra_nn/dataset_builder.py \
    --data-dir "$DATA/training_data" \
    --out "$OUT" \
    --gate

echo "Dataset rebuild complete"

# Verify parent features exist
python -c "
import h5py
with h5py.File('$OUT', 'r') as f:
    mol = list(f.keys())[0]
    grp = f[mol]
    assert 'parent_rho0_s' in grp, 'parent_rho0_s missing'
    assert 'parent_vnuc_s' in grp, 'parent_vnuc_s missing'
    print(f'Verified: {mol}/parent_rho0_s shape={grp[\"parent_rho0_s\"].shape}')
    print(f'Verified: {mol}/parent_vnuc_s shape={grp[\"parent_vnuc_s\"].shape}')
"
