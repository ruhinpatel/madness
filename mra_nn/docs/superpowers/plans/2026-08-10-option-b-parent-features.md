# Option B: Parent Node Features Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Model:** Sonnet 4.6 for Tasks 1-3, Opus 4.6 for Task 4 (model architecture change).

**Goal:** Add parent node rho0/vnuc s-coefficients as model input features so the network has cross-level context at coarse levels (1-9), where SCF convergence is determined.

**Architecture:** For each sample, extract the parent node's rho0 and vnuc s-coefficients (Key(n-1, l//2) via pymra). Store as two new HDF5 fields. Concatenate the 1024 parent floats to the trunk input (1792 -> 2816). Level 0 nodes get zero-padded parent features. Train from scratch.

**Tech Stack:** Python (PyTorch, pymra, h5py), SLURM (A100 GPU), MADNESS C++ (existing binaries for SCF test)

## Global Constraints

- `PYTHONPATH` must include `/gpfs/projects/rjh/ruhin/madness-ruhin` and `/gpfs/projects/rjh/adrian/pymra/src`
- Virtual env: `/gpfs/projects/rjh/ruhin/mra_nn/.venv/`
- Working data dir: `/gpfs/projects/rjh/ruhin/mra_nn/` (training_data/, checkpoints/, logs/)
- Code repo: `/gpfs/projects/rjh/ruhin/madness-ruhin/` (branch `feat/mra-nn-data`)
- k=8, ndim=3, k^3=512
- Never add `Co-Authored-By: Claude` lines to commit messages
- `MAD_NUM_THREADS` must always be set to `ntasks - 1`
- Checkpoint incompatible with Option A (first trunk layer shape changes) — train from scratch
- Parent key: `key.parent()` returns `Key(n-1, tuple(li >> 1 for li in l))`. At level 0 (n=0), there is no parent — zero-pad.
- `node_s(tree, key)` reconstructs s-coefficients at any node in the tree via the two-scale relation, even if that node is not a leaf.

---

### Task 1: Add Parent Features to Dataset Builder

**Files:**
- Modify: `mra_nn/dataset_builder.py:53-154` (HALO_OFFSETS area, `process_molecule()`, `_append()`)
- Modify: `mra_nn/dataset_builder.py:157-168` (`write_molecule()` — add new datasets)

**Interfaces:**
- Consumes: pymra `Key.parent()`, `node_s(tree, key)` — both exist, no changes needed
- Produces: Two new HDF5 datasets per molecule group: `parent_rho0_s [N, 512] float32`, `parent_vnuc_s [N, 512] float32`. Consumed by Task 2 (dataset.py) and Task 3 (rebuild).

- [ ] **Step 1: Add parent feature extraction to `_append()` in `process_molecule()`**

In `mra_nn/dataset_builder.py`, add two new accumulator lists after line 101 and modify `_append()` to extract parent features:

```python
    # After existing list declarations (line 101):
    parent_rho0_s_list = []
    parent_vnuc_s_list = []

    def _append(key: Key, rho_s: np.ndarray, log_d: float, refine: int, neg: int):
        rho0_s = node_s(rho0, key)
        vnuc_s = node_s(vnuc, key)
        h_rho0 = np.stack([safe_node_s(rho0, hk, k, ndim).ravel()
                           for hk in halo_keys(key)])          # [6, k^3]
        h_vnuc     = np.stack([safe_node_s(vnuc, hk, k, ndim).ravel()
                               for hk in halo_keys(key)])
        rho_s_flat = rho_s.ravel()

        # Parent features: Key(n-1, l//2). Zero-pad at level 0.
        if key.n > 0:
            parent_key = key.parent()
            p_rho0 = node_s(rho0, parent_key).ravel()
            p_vnuc = node_s(vnuc, parent_key).ravel()
        else:
            p_rho0 = np.zeros(k ** ndim, dtype=np.float64)
            p_vnuc = np.zeros(k ** ndim, dtype=np.float64)

        rho0_s_list.append(rho0_s.ravel())
        vnuc_s_list.append(vnuc_s.ravel())
        halo_rho0_list.append(h_rho0)
        halo_vnuc_list.append(h_vnuc)
        rho_s_list.append(rho_s_flat)
        log_dnorm_list.append(log_d)
        refine_list.append(refine)
        level_list.append(key.n)
        l_trans_list.append(list(key.l))
        negative_list.append(neg)
        parent_rho0_s_list.append(p_rho0)
        parent_vnuc_s_list.append(p_vnuc)
```

- [ ] **Step 2: Add parent arrays to the returned dict**

At the end of `process_molecule()`, add the two new arrays to the return dict (after the existing `"negative"` entry):

```python
    return {
        "rho0_s":    np.array(rho0_s_list,    dtype=np.float32),
        "vnuc_s":    np.array(vnuc_s_list,    dtype=np.float32),
        "halo_rho0": np.array(halo_rho0_list, dtype=np.float32),
        "halo_vnuc": np.array(halo_vnuc_list, dtype=np.float32),
        "rho_s":     np.array(rho_s_list,     dtype=np.float32),
        "log_dnorm": np.array(log_dnorm_list, dtype=np.float32),
        "refine":    np.array(refine_list,    dtype=np.int8),
        "level":     np.array(level_list,     dtype=np.int8),
        "l_trans":   np.array(l_trans_list,   dtype=np.int32),
        "negative":  np.array(negative_list,  dtype=np.int8),
        "parent_rho0_s": np.array(parent_rho0_s_list, dtype=np.float32),
        "parent_vnuc_s": np.array(parent_vnuc_s_list, dtype=np.float32),
    }
```

- [ ] **Step 3: Verify with a single-molecule dry run**

```bash
cd /gpfs/projects/rjh/ruhin/madness-ruhin
source /gpfs/projects/rjh/ruhin/mra_nn/.venv/bin/activate
export PYTHONPATH=/gpfs/projects/rjh/ruhin/madness-ruhin:/gpfs/projects/rjh/adrian/pymra/src:${PYTHONPATH:-}

python mra_nn/dataset_builder.py \
    --data-dir /gpfs/projects/rjh/ruhin/mra_nn/training_data \
    --out /tmp/test_parent_features.h5 \
    --mols h2o \
    --gate
```

Expected: gate PASS, and the output should include `parent_rho0_s` and `parent_vnuc_s` datasets. Verify:

```bash
python -c "
import h5py
with h5py.File('/tmp/test_parent_features.h5', 'r') as f:
    grp = f['h2o']
    for name in sorted(grp.keys()):
        print(f'  {name}: {grp[name].shape} {grp[name].dtype}')
    # Verify level-0 samples have zero parent features
    import numpy as np
    levels = grp['level'][:]
    parent_rho0 = grp['parent_rho0_s'][:]
    l0_mask = (levels == 0)
    if l0_mask.any():
        assert np.allclose(parent_rho0[l0_mask], 0.0), 'Level 0 parent should be zeros'
        print(f'  Level 0 zero-pad check: PASS ({l0_mask.sum()} samples)')
    # Verify non-zero parent features exist at other levels
    non_l0 = parent_rho0[~l0_mask]
    assert np.any(non_l0 != 0), 'Non-level-0 parent features should be non-zero'
    print(f'  Non-level-0 non-zero check: PASS')
"
```

Expected output includes `parent_rho0_s: (N, 512) float32` and `parent_vnuc_s: (N, 512) float32`, both checks PASS.

- [ ] **Step 4: Clean up temp file and commit**

```bash
rm /tmp/test_parent_features.h5
cd /gpfs/projects/rjh/ruhin/madness-ruhin
git add mra_nn/dataset_builder.py
git commit -m "feat(mra-nn): add parent rho0/vnuc s-coefficients to dataset builder

Extract parent node s-coefficients via key.parent() for cross-level
context. Level 0 nodes get zero-padded parent features. Gate check
still passes."
```

---

### Task 2: Update Dataset and Training Loop for Parent Features

**Files:**
- Modify: `mra_nn/dataset.py:27-29` (`FIELD_NAMES` list)
- Modify: `mra_nn/configs/single_task.yaml:67-78` (model section)
- Modify: `mra_nn/train.py:57-61` (model forward call in `train_one_epoch`)
- Modify: `mra_nn/train.py:106-110` (model forward call in `evaluate`)

**Interfaces:**
- Consumes: HDF5 fields `parent_rho0_s`, `parent_vnuc_s` from Task 1
- Produces: `batch["parent_rho0_s"]` and `batch["parent_vnuc_s"]` tensors [B, 512] passed to model forward. Config key `model.use_parent_features: true`. Consumed by Task 4 (model changes).

- [ ] **Step 1: Add parent fields to `MRADataset.FIELD_NAMES`**

In `mra_nn/dataset.py`, modify the FIELD_NAMES list (line 27):

```python
    FIELD_NAMES = [
        "rho0_s", "vnuc_s", "halo_rho0", "halo_vnuc",
        "rho_s", "log_dnorm", "refine", "level", "negative",
        "parent_rho0_s", "parent_vnuc_s",
    ]
```

- [ ] **Step 2: Add `use_parent_features` to config**

In `mra_nn/configs/single_task.yaml`, add to the model section (after `single_task: true`):

```yaml
  single_task: true
  use_parent_features: true
```

- [ ] **Step 3: Pass parent features to model in `train_one_epoch()`**

In `mra_nn/train.py`, modify the model forward call in `train_one_epoch()` (lines 57-61):

```python
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            forward_args = [
                batch["rho0_s"], batch["vnuc_s"],
                batch["halo_rho0"], batch["halo_vnuc"],
                batch["level"],
            ]
            if "parent_rho0_s" in batch:
                forward_args.extend([batch["parent_rho0_s"], batch["parent_vnuc_s"]])
            rs, ld, ref = model(*forward_args)
            if single_task:
                total_loss, components = loss_fn(batch, rs)
            else:
                total_loss, components = loss_fn(batch, rs, ld, ref)
```

- [ ] **Step 4: Pass parent features to model in `evaluate()`**

In `mra_nn/train.py`, modify the model forward call in `evaluate()` (lines 106-110). Same pattern:

```python
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            forward_args = [
                batch["rho0_s"], batch["vnuc_s"],
                batch["halo_rho0"], batch["halo_vnuc"],
                batch["level"],
            ]
            if "parent_rho0_s" in batch:
                forward_args.extend([batch["parent_rho0_s"], batch["parent_vnuc_s"]])
            rs, ld, ref = model(*forward_args)
            if single_task:
                total_loss, components = loss_fn(batch, rs)
            else:
                total_loss, components = loss_fn(batch, rs, ld, ref)
```

- [ ] **Step 5: Verify dataset loads with parent features**

```bash
cd /gpfs/projects/rjh/ruhin/madness-ruhin
source /gpfs/projects/rjh/ruhin/mra_nn/.venv/bin/activate
export PYTHONPATH=/gpfs/projects/rjh/ruhin/madness-ruhin:/gpfs/projects/rjh/adrian/pymra/src:${PYTHONPATH:-}

python -c "
from mra_nn.dataset import MRADataset
ds = MRADataset('/gpfs/projects/rjh/ruhin/mra_nn/training_dataset.h5', ['h2o'])
sample = ds[0]
assert 'parent_rho0_s' in sample, 'parent_rho0_s not in sample'
assert 'parent_vnuc_s' in sample, 'parent_vnuc_s not in sample'
assert sample['parent_rho0_s'].shape == (512,), f'wrong shape: {sample[\"parent_rho0_s\"].shape}'
print(f'Dataset loaded: {len(ds)} samples, parent features present')
print(f'parent_rho0_s shape: {sample[\"parent_rho0_s\"].shape}')
"
```

Expected: prints shapes confirming [512] parent feature tensors.

**Note:** This test will fail until Task 3 rebuilds the dataset with parent features. If testing before Task 3, use the temp h5 file from Task 1's dry run:
```bash
python -c "
from mra_nn.dataset import MRADataset
ds = MRADataset('/tmp/test_parent_features.h5', ['h2o'])
..."
```

- [ ] **Step 6: Commit**

```bash
cd /gpfs/projects/rjh/ruhin/madness-ruhin
git add mra_nn/dataset.py mra_nn/train.py mra_nn/configs/single_task.yaml
git commit -m "feat(mra-nn): wire parent features through dataset and training loop

MRADataset loads parent_rho0_s and parent_vnuc_s fields.
Training loop passes them to model forward when present.
Config adds use_parent_features: true."
```

---

### Task 3: Rebuild Dataset with Parent Features

**Files:**
- Create: `mra_nn/slurm/rebuild_dataset_b.sh`
- No code changes (uses dataset_builder.py from Task 1)

**Interfaces:**
- Consumes: `dataset_builder.py` with parent features (Task 1)
- Produces: `/gpfs/projects/rjh/ruhin/mra_nn/training_dataset.h5` with parent_rho0_s/parent_vnuc_s fields. Consumed by Task 5 (training).

- [ ] **Step 1: Create SLURM rebuild script**

Create `mra_nn/slurm/rebuild_dataset_b.sh`:

```bash
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
```

- [ ] **Step 2: Validate and submit**

```bash
/cm/shared/apps/slurm/21.08.8/bin/sbatch --test-only /gpfs/projects/rjh/ruhin/madness-ruhin/mra_nn/slurm/rebuild_dataset_b.sh
```

Expected: `Job NNNNN to start at ...` (no errors).

Then submit:

```bash
/cm/shared/apps/slurm/21.08.8/bin/sbatch /gpfs/projects/rjh/ruhin/madness-ruhin/mra_nn/slurm/rebuild_dataset_b.sh
```

Monitor:
```bash
squeue -u ruhipatel
```

Job takes ~1-2 hours for 51 molecules. Gate check verifies rho_s integrity.

- [ ] **Step 3: Verify rebuild output**

Once job completes, check the log:

```bash
tail -20 /gpfs/projects/rjh/ruhin/mra_nn/logs/rebuild_b_<jobid>.out
```

Expected: `gate [<mol>] PASS`, `Dataset rebuild complete`, and parent feature verification lines.

Spot-check the dataset:

```bash
cd /gpfs/projects/rjh/ruhin/madness-ruhin
source /gpfs/projects/rjh/ruhin/mra_nn/.venv/bin/activate
export PYTHONPATH=/gpfs/projects/rjh/ruhin/madness-ruhin:/gpfs/projects/rjh/adrian/pymra/src:${PYTHONPATH:-}

python -c "
import h5py, numpy as np
with h5py.File('/gpfs/projects/rjh/ruhin/mra_nn/training_dataset.h5', 'r') as f:
    total = 0
    for mol in f.keys():
        grp = f[mol]
        n = grp['level'].shape[0]
        total += n
        assert 'parent_rho0_s' in grp, f'{mol}: parent_rho0_s missing'
        assert grp['parent_rho0_s'].shape == (n, 512), f'{mol}: wrong shape'
    print(f'Total samples: {total}')
    print(f'All molecules have parent features')
"
```

Expected: ~207k samples, all molecules have parent features.

- [ ] **Step 4: Commit SLURM script**

```bash
cd /gpfs/projects/rjh/ruhin/madness-ruhin
git add mra_nn/slurm/rebuild_dataset_b.sh
git commit -m "feat(mra-nn): add SLURM script for Option B dataset rebuild

Rebuilds training_dataset.h5 with parent_rho0_s and parent_vnuc_s
fields. Backs up existing dataset. Runs gate check and verifies
parent features exist."
```

---

### Task 4: Add Parent Features to MRANet Model

**Files:**
- Modify: `mra_nn/model.py:119-226` (MRANet class — forward signature, trunk input)
- Modify: `mra_nn/model.py:229-243` (`build_model()` — pass new config flag)

**Interfaces:**
- Consumes: `use_parent_features: bool` from config (Task 2), `parent_rho0_s` / `parent_vnuc_s` tensors [B, 512] from dataset (Task 2)
- Produces: MRANet.forward() accepts optional `parent_rho0_s` and `parent_vnuc_s` kwargs. Trunk input dimension adjusts automatically. Consumed by Task 5 (training) and predict.py (inference).

- [ ] **Step 1: Add `use_parent_features` parameter to MRANet.__init__()**

In `mra_nn/model.py`, modify the `__init__` signature (line 128) and trunk input computation (lines 156-159):

```python
    def __init__(
        self,
        k_cubed: int = 512,
        n_faces: int = 6,
        n_levels: int = 19,
        level_embed_dim: int = 32,
        face_embed_dim: int = 8,
        halo_encoder_hidden: int = 256,
        halo_encoder_out: int = 128,
        trunk_dims: tuple = (1024, 512, 256),
        dropout: float = 0.1,
        single_task: bool = False,
        use_parent_features: bool = False,
    ) -> None:
        super().__init__()
        self.use_parent_features = use_parent_features

        # --- Halo encoder (shared across 6 faces) ---
        self.halo_encoder = HaloEncoder(
            k_cubed=k_cubed,
            face_embed_dim=face_embed_dim,
            hidden_dim=halo_encoder_hidden,
            out_dim=halo_encoder_out,
            n_faces=n_faces,
        )

        # --- Level embedding (for FiLM conditioning) ---
        self.level_embedding = nn.Embedding(n_levels, level_embed_dim)

        # --- Trunk MLP with FiLM conditioning ---
        center_dim = 2 * k_cubed  # rho0_s + vnuc_s
        parent_dim = 2 * k_cubed if use_parent_features else 0
        halo_out_dim = n_faces * halo_encoder_out
        trunk_input_dim = halo_out_dim + center_dim + parent_dim

        trunk_layers = []
        in_dim = trunk_input_dim
        for out_dim in trunk_dims:
            trunk_layers.append(
                FiLMLayer(in_dim, out_dim, level_embed_dim, dropout)
            )
            in_dim = out_dim
        self.trunk = nn.ModuleList(trunk_layers)

        # --- Output heads ---
        final_dim = trunk_dims[-1]  # 256
        self.single_task = single_task
        self.head_rho_s = nn.Linear(final_dim, k_cubed)
        if not single_task:
            self.head_log_dnorm = nn.Linear(final_dim, 1)
            nn.init.constant_(self.head_log_dnorm.bias, -27.5)
            self.head_refine = nn.Linear(final_dim, 1)
```

- [ ] **Step 2: Update MRANet.forward() to accept and use parent features**

Modify the forward method (line 179):

```python
    def forward(
        self,
        rho0_s: torch.Tensor,
        vnuc_s: torch.Tensor,
        halo_rho0: torch.Tensor,
        halo_vnuc: torch.Tensor,
        level: torch.Tensor,
        parent_rho0_s: torch.Tensor | None = None,
        parent_vnuc_s: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        rho0_s         : [B, k^3]
        vnuc_s         : [B, k^3]
        halo_rho0      : [B, 6, k^3]
        halo_vnuc      : [B, 6, k^3]
        level          : [B] (long)
        parent_rho0_s  : [B, k^3] or None (required if use_parent_features=True)
        parent_vnuc_s  : [B, k^3] or None (required if use_parent_features=True)

        Returns
        -------
        rho_s        : [B, k^3]
        log_dnorm    : [B]
        refine_logit : [B]
        """
        # Halo encoding
        halo_emb = self.halo_encoder(halo_rho0, halo_vnuc)  # [B, 768]

        # Center features
        center = torch.cat([rho0_s, vnuc_s], dim=-1)  # [B, 1024]

        # Level embedding (for FiLM, not concatenated)
        level_emb = self.level_embedding(level)  # [B, 32]

        # Trunk input
        if self.use_parent_features:
            parent = torch.cat([parent_rho0_s, parent_vnuc_s], dim=-1)  # [B, 1024]
            x = torch.cat([halo_emb, center, parent], dim=-1)  # [B, 2816]
        else:
            x = torch.cat([halo_emb, center], dim=-1)  # [B, 1792]

        # FiLM-conditioned trunk
        for film_layer in self.trunk:
            x = film_layer(x, level_emb)

        # Output heads
        rho_s = self.head_rho_s(x) + rho0_s  # [B, 512]
        if self.single_task:
            return rho_s, None, None
        log_dnorm = self.head_log_dnorm(x).squeeze(-1)  # [B]
        refine_logit = self.head_refine(x).squeeze(-1)  # [B]

        return rho_s, log_dnorm, refine_logit
```

- [ ] **Step 3: Update `build_model()` to pass `use_parent_features`**

Modify `build_model()` (line 229):

```python
def build_model(cfg: dict) -> MRANet:
    """Construct MRANet from config dict."""
    m = cfg["model"]
    return MRANet(
        k_cubed=m["k_cubed"],
        n_faces=m["n_faces"],
        n_levels=m["n_levels"],
        level_embed_dim=m["level_embed_dim"],
        face_embed_dim=m["face_embed_dim"],
        halo_encoder_hidden=m["halo_encoder_hidden"],
        halo_encoder_out=m["halo_encoder_out"],
        trunk_dims=tuple(m["trunk_dims"]),
        dropout=m["dropout"],
        single_task=m.get("single_task", False),
        use_parent_features=m.get("use_parent_features", False),
    )
```

- [ ] **Step 4: Verify model builds and forward pass works**

```bash
cd /gpfs/projects/rjh/ruhin/madness-ruhin
source /gpfs/projects/rjh/ruhin/mra_nn/.venv/bin/activate
export PYTHONPATH=/gpfs/projects/rjh/ruhin/madness-ruhin:/gpfs/projects/rjh/adrian/pymra/src:${PYTHONPATH:-}

python -c "
import torch
from mra_nn.model import MRANet

# Without parent features (backward compat)
model_old = MRANet(k_cubed=512, single_task=True, use_parent_features=False)
n_old = sum(p.numel() for p in model_old.parameters())
x = model_old(
    torch.randn(2, 512), torch.randn(2, 512),
    torch.randn(2, 6, 512), torch.randn(2, 6, 512),
    torch.tensor([5, 10]),
)
print(f'Without parent: {n_old:,} params, output shape {x[0].shape}')

# With parent features
model_new = MRANet(k_cubed=512, single_task=True, use_parent_features=True)
n_new = sum(p.numel() for p in model_new.parameters())
x = model_new(
    torch.randn(2, 512), torch.randn(2, 512),
    torch.randn(2, 6, 512), torch.randn(2, 6, 512),
    torch.tensor([5, 10]),
    parent_rho0_s=torch.randn(2, 512),
    parent_vnuc_s=torch.randn(2, 512),
)
print(f'With parent: {n_new:,} params, output shape {x[0].shape}')
print(f'Param increase: {n_new - n_old:,} ({(n_new - n_old)/n_old*100:.1f}%)')
assert x[0].shape == (2, 512), f'wrong output shape: {x[0].shape}'
print('Forward pass OK')
"
```

Expected: ~3.04M without parent, ~4.09M with parent, output shape (2, 512), forward pass OK.

- [ ] **Step 5: Commit**

```bash
cd /gpfs/projects/rjh/ruhin/madness-ruhin
git add mra_nn/model.py
git commit -m "feat(mra-nn): add parent feature support to MRANet

New use_parent_features flag widens trunk input from 1792 to 2816.
Parent rho0/vnuc s-coefficients (1024 floats) concatenated alongside
center and halo features. Backward compatible: flag defaults to False."
```

---

### Task 5: Update Inference and Submit Training

**Files:**
- Modify: `mra_nn/predict.py:63-97` (`_extract_features()` — add parent extraction)
- Modify: `mra_nn/predict.py:203-211` (`predict_density_simple()` — pass parent to model)
- Create: `mra_nn/slurm/train_option_b.sh`

**Interfaces:**
- Consumes: MRANet with `use_parent_features=True` (Task 4), rebuilt dataset (Task 3)
- Produces: Trained checkpoint at `/gpfs/projects/rjh/ruhin/mra_nn/checkpoints/<date>/best.pt`. Density predictions via `predict.py` with parent features.

- [ ] **Step 1: Add parent feature extraction to `_extract_features()`**

In `mra_nn/predict.py`, modify `_extract_features()` (lines 63-97) to optionally extract parent features:

```python
def _extract_features(
    keys: List[Key],
    rho0_tree: FunctionTree,
    vnuc_tree: FunctionTree,
    device: torch.device,
    include_parent: bool = False,
) -> dict:
    """Extract model input features for a batch of keys."""
    k = rho0_tree.k
    ndim = rho0_tree.ndim

    rho0_s_list = []
    vnuc_s_list = []
    halo_rho0_list = []
    halo_vnuc_list = []
    level_list = []
    parent_rho0_list = []
    parent_vnuc_list = []

    for key in keys:
        rho0_s_list.append(node_s(rho0_tree, key).ravel())
        vnuc_s_list.append(node_s(vnuc_tree, key).ravel())

        h_rho0 = np.stack([_safe_node_s(rho0_tree, hk).ravel()
                           for hk in _halo_keys(key)])
        h_vnuc = np.stack([_safe_node_s(vnuc_tree, hk).ravel()
                           for hk in _halo_keys(key)])
        halo_rho0_list.append(h_rho0)
        halo_vnuc_list.append(h_vnuc)
        level_list.append(key.n)

        if include_parent:
            if key.n > 0:
                parent_key = key.parent()
                parent_rho0_list.append(node_s(rho0_tree, parent_key).ravel())
                parent_vnuc_list.append(node_s(vnuc_tree, parent_key).ravel())
            else:
                parent_rho0_list.append(np.zeros(k ** ndim, dtype=np.float64))
                parent_vnuc_list.append(np.zeros(k ** ndim, dtype=np.float64))

    result = {
        "rho0_s": torch.from_numpy(np.array(rho0_s_list, dtype=np.float32)).to(device),
        "vnuc_s": torch.from_numpy(np.array(vnuc_s_list, dtype=np.float32)).to(device),
        "halo_rho0": torch.from_numpy(np.array(halo_rho0_list, dtype=np.float32)).to(device),
        "halo_vnuc": torch.from_numpy(np.array(halo_vnuc_list, dtype=np.float32)).to(device),
        "level": torch.tensor(level_list, dtype=torch.long).to(device),
    }
    if include_parent:
        result["parent_rho0_s"] = torch.from_numpy(
            np.array(parent_rho0_list, dtype=np.float32)
        ).to(device)
        result["parent_vnuc_s"] = torch.from_numpy(
            np.array(parent_vnuc_list, dtype=np.float32)
        ).to(device)
    return result
```

- [ ] **Step 2: Update `predict_density_simple()` to pass parent features**

In `mra_nn/predict.py`, modify the batch loop in `predict_density_simple()` (lines 203-211):

```python
    # Detect if model uses parent features
    use_parent = getattr(model, 'use_parent_features', False)

    # Predict in batches
    for start in range(0, len(leaf_keys), batch_size):
        batch_keys = leaf_keys[start : start + batch_size]
        features = _extract_features(
            batch_keys, rho0_tree, vnuc_tree, device,
            include_parent=use_parent,
        )
        forward_args = [
            features["rho0_s"], features["vnuc_s"],
            features["halo_rho0"], features["halo_vnuc"],
            features["level"],
        ]
        if use_parent:
            forward_args.extend([features["parent_rho0_s"], features["parent_vnuc_s"]])
        rho_s, _, _ = model(*forward_args)
        rho_s_np = rho_s.cpu().numpy()
        for i, key in enumerate(batch_keys):
            if use_model_levels is not None and key.n not in use_model_levels:
                predicted_tree.nodes[key] = Node(
                    s=node_s(rho0_tree, key).copy()
                )
            else:
                predicted_tree.nodes[key] = Node(
                    s=rho_s_np[i].reshape((k,) * ndim).astype(np.float64)
                )
```

- [ ] **Step 3: Create training SLURM script**

Create `mra_nn/slurm/train_option_b.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=mra-nn-b
#SBATCH --partition=a100-long
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --output=/gpfs/projects/rjh/ruhin/mra_nn/logs/train_b_%j.out
#SBATCH --error=/gpfs/projects/rjh/ruhin/mra_nn/logs/train_b_%j.err

set -euo pipefail

source /gpfs/projects/rjh/ruhin/mra_nn/.venv/bin/activate
export PYTHONPATH=/gpfs/projects/rjh/ruhin/madness-ruhin:/gpfs/projects/rjh/adrian/pymra/src:${PYTHONPATH:-}

python -m mra_nn.train --config /gpfs/projects/rjh/ruhin/madness-ruhin/mra_nn/configs/single_task.yaml
```

- [ ] **Step 4: Validate and submit training job**

```bash
/cm/shared/apps/slurm/21.08.8/bin/sbatch --test-only /gpfs/projects/rjh/ruhin/madness-ruhin/mra_nn/slurm/train_option_b.sh
```

Expected: `Job NNNNN to start at ...`

Then submit:
```bash
/cm/shared/apps/slurm/21.08.8/bin/sbatch /gpfs/projects/rjh/ruhin/madness-ruhin/mra_nn/slurm/train_option_b.sh
```

Training takes ~2-4 hours on A100 (120 epochs, ~207k samples, batch 4096).

- [ ] **Step 5: Commit**

```bash
cd /gpfs/projects/rjh/ruhin/madness-ruhin
git add mra_nn/predict.py mra_nn/slurm/train_option_b.sh
git commit -m "feat(mra-nn): add parent features to inference and Option B training script

predict.py extracts parent s-coefficients at inference time when model
has use_parent_features=True. Training script for Option B on A100."
```

---

### Task 6: Evaluate and Run SCF Test

**Files:**
- No new code (uses existing `diagnose_option_a.py`, `compare_densities.py`, `slurm/scf_test.sh`)
- Modify: `mra_nn/slurm/scf_test.sh` (update checkpoint path to Option B)

**Interfaces:**
- Consumes: Trained Option B checkpoint from Task 5
- Produces: Per-level diagnostic, SCF iteration comparison, density comparison

- [ ] **Step 1: Run per-level diagnostic on Option B checkpoint**

Once training completes, find the checkpoint:
```bash
ls -lt /gpfs/projects/rjh/ruhin/mra_nn/checkpoints/ | head -5
```

Run the diagnostic (substitute `<date>` with the actual checkpoint directory):
```bash
cd /gpfs/projects/rjh/ruhin/madness-ruhin
source /gpfs/projects/rjh/ruhin/mra_nn/.venv/bin/activate
export PYTHONPATH=/gpfs/projects/rjh/ruhin/madness-ruhin:/gpfs/projects/rjh/adrian/pymra/src:${PYTHONPATH:-}

python /gpfs/projects/rjh/ruhin/mra_nn/diagnose_option_a.py \
    --checkpoint /gpfs/projects/rjh/ruhin/mra_nn/checkpoints/<date>/best.pt \
    --dataset /gpfs/projects/rjh/ruhin/mra_nn/training_dataset.h5
```

**Key metrics to check:**
- Levels 1-9: any ratio < 1.0? (currently all at parity — parent features should improve these)
- Levels 10-14: still < 1.0? (should remain good or improve)
- Level 0: still garbage? (expected — still masked, still zero-padded)

- [ ] **Step 2: Update scf_test.sh with Option B checkpoint**

In `mra_nn/slurm/scf_test.sh`, update the checkpoint path and optionally adjust `--use-model-levels` based on diagnostic results:

```bash
# Update this line with the Option B checkpoint path:
$PYTHON mra_nn/predict.py \
    --checkpoint "$DATA/checkpoints/<date>/best.pt" \
    --rho0 "$DATA/training_data/ch3oh/rho0.mad.h5" \
    --vnuc "$DATA/training_data/ch3oh/vnuc.mad.h5" \
    --n-electrons 18 \
    --use-model-levels "1,2,3,4,5,6,7,8,9,10,11,12,13,14" \
    --out "$TESTDIR/rhoML.mad.h5"
```

The `--use-model-levels` range should be expanded based on diagnostic results. If levels 1-9 now beat baseline, include them. Always exclude level 0.

- [ ] **Step 3: Submit SCF test**

```bash
/cm/shared/apps/slurm/21.08.8/bin/sbatch /gpfs/projects/rjh/ruhin/madness-ruhin/mra_nn/slurm/scf_test.sh
```

Job takes ~30-60 min. Monitor with `squeue -u ruhipatel`.

- [ ] **Step 4: Analyze results**

Once job completes:
```bash
cat /gpfs/projects/rjh/ruhin/mra_nn/logs/scf_test_<jobid>.out
```

Compare against baseline:

| Metric | Baseline (rho0) | Option A clamped [10-14] | Option B clamped [1-14] |
|--------|-----------------|--------------------------|--------------------------|
| Energy (Ha) | -114.85038034 | -114.85038032 | ? |
| Protocol 1 iters | 9 | 9 | ? |
| Protocol 2 iters | 3 | 3 | ? |
| Total iters | 12 | 12 | ? |

**Success:** Energy within 1e-3 Ha AND total iterations < 12.
**Neutral:** Same state, same iterations (parent features didn't help coarse levels enough).
**Failure:** Wrong state (investigate).

- [ ] **Step 5: Update project documentation**

Update `mra_nn/CLAUDE.md` and `mra_nn/docs/2026-08-10-scf-test-postmortem.md` with Option B results. Add Decision 22 to the post-mortem with training outcome and SCF results.

```bash
cd /gpfs/projects/rjh/ruhin/madness-ruhin
git add mra_nn/CLAUDE.md mra_nn/docs/2026-08-10-scf-test-postmortem.md mra_nn/slurm/scf_test.sh
git commit -m "docs(mra-nn): Option B training results and SCF test

<fill in with actual results>"
```

- [ ] **Step 6: Decision gate**

Based on results:
- **Levels 1-7 improved + fewer SCF iterations:** Option B succeeded. Consider further improvements (more molecules, deeper parent chain).
- **Levels 1-7 improved but same SCF iterations:** Improvement isn't enough to cross the convergence threshold. May need stronger multi-scale signal.
- **Levels 1-7 still at parity:** Parent features alone insufficient. Correction requires broader context. Consider message-passing (GNN on tree) or pivot to tree structure prediction (Approach C).
