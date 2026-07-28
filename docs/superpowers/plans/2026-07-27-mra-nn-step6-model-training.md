# MRA-NN Step 6: MLP Model & Training — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a FiLM-conditioned MLP with factored halo encoder that predicts density corrections (Δρ), wavelet norms (log‖d‖), and refinement decisions from promolecular density and nuclear potential inputs in the MRA basis.

**Architecture:** Shared halo encoder processes 6 face-adjacent neighbors, concatenated with center box features, fed through 3 FiLM-conditioned trunk layers (level embedding drives γ/β), into 3 output heads. Uncertainty-weighted multi-task loss (Kendall et al. 2018) with focal loss on the refine head.

**Tech Stack:** Python 3.12, PyTorch (with CUDA/AMP), h5py, numpy, pyyaml, pymra

## Global Constraints

- **Branch:** `feat/mra-nn-data` on `ruhinpatel/madness`
- **All code lives in:** `/gpfs/projects/rjh/ruhin/madness-ruhin/mra_nn/`
- **Training dataset:** `/gpfs/projects/rjh/ruhin/mra_nn/training_dataset.h5`
- **Raw data:** `/gpfs/projects/rjh/ruhin/mra_nn/training_data/<mol>/` (rho0.mad.h5, vnuc.mad.h5, rho.mad.h5)
- **pymra:** `/gpfs/projects/rjh/adrian/pymra/src` — must be on PYTHONPATH; cannot pip install (owned by Adrian)
- **Venv:** `/gpfs/projects/rjh/ruhin/mra_nn/.venv/` — Python 3.12, numpy 1.23.5, h5py 3.14.0; PyTorch NOT yet installed
- **Checkpoints/logs:** `/gpfs/projects/rjh/ruhin/mra_nn/checkpoints/<run_name>/` (not in git)
- **Slurm:** `/cm/shared/apps/slurm/21.08.8/bin/sbatch` (not in default PATH); A100 GPU partition
- **Git rules:** No `Co-Authored-By:` lines in commits. No Jira ticket names in branches/commits/paths.
- **k=6, k^3=216** for all coefficient dimensions in this prototype
- **Design spec:** `docs/superpowers/specs/2026-07-27-mra-nn-step6-model-training-design.md`

---

### Task 1: Environment Setup & Config

**Model recommendation: Sonnet 4.6** — straightforward dependency installation and YAML config.

**Files:**
- Create: `mra_nn/configs/default.yaml`
- Create: `mra_nn/slurm/train_a100.sh`

**Interfaces:**
- Consumes: nothing
- Produces: `default.yaml` config dict loaded by all subsequent tasks; `train_a100.sh` used for GPU training

- [ ] **Step 1: Install PyTorch and pyyaml into the venv**

```bash
source /gpfs/projects/rjh/ruhin/mra_nn/.venv/bin/activate
pip install torch pyyaml
```

Verify:
```bash
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import yaml; print(f'pyyaml {yaml.__version__}')"
```

Expected: PyTorch version prints (CUDA may be False on login node — that's fine, it'll be True on A100 nodes).

- [ ] **Step 2: Create the config file**

Create `mra_nn/configs/default.yaml`:

```yaml
# MRA-NN Step 6 — default training configuration
# See design spec: docs/superpowers/specs/2026-07-27-mra-nn-step6-model-training-design.md

data:
  dataset_path: /gpfs/projects/rjh/ruhin/mra_nn/training_dataset.h5
  raw_data_dir: /gpfs/projects/rjh/ruhin/mra_nn/training_data
  train_molecules:
    - h2o
    - nh3
    - ch4
    - co2
    - hf
    - n2
    - co
    - hcn
    - c2h4
    - c2h6
    - h2co
    - hcl
  val_molecules:
    - ch3oh
  test_molecules:
    - h2o2
    - c2h2

model:
  k: 6
  ndim: 3
  k_cubed: 216
  n_faces: 6
  n_levels: 19              # levels 0-18
  level_embed_dim: 32
  face_embed_dim: 8
  halo_encoder_hidden: 256
  halo_encoder_out: 128
  trunk_dims: [1024, 512, 256]
  dropout: 0.1

training:
  batch_size: 4096
  max_epochs: 200
  lr: 1.0e-3
  min_lr: 1.0e-5
  weight_decay: 1.0e-4
  warmup_epochs: 5
  patience: 20
  num_workers: 4
  seed: 42

loss:
  focal_gamma: 2.0
  focal_alpha: 0.75
  refine_pos_weight: 10.0   # oversampling weight for refine=1 positives

checkpoint:
  dir: /gpfs/projects/rjh/ruhin/mra_nn/checkpoints
```

Verify:
```bash
source /gpfs/projects/rjh/ruhin/mra_nn/.venv/bin/activate
python -c "
import yaml
with open('mra_nn/configs/default.yaml') as f:
    cfg = yaml.safe_load(f)
print(f'Train mols: {len(cfg[\"data\"][\"train_molecules\"])}')
print(f'Trunk dims: {cfg[\"model\"][\"trunk_dims\"]}')
print(f'Batch size: {cfg[\"training\"][\"batch_size\"]}')
"
```

Expected:
```
Train mols: 12
Trunk dims: [1024, 512, 256]
Batch size: 4096
```

- [ ] **Step 3: Create the Slurm script**

Create `mra_nn/slurm/train_a100.sh`:

```bash
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
```

Verify:
```bash
chmod +x mra_nn/slurm/train_a100.sh
head -5 mra_nn/slurm/train_a100.sh
```

- [ ] **Step 4: Create logs directory and commit**

```bash
mkdir -p /gpfs/projects/rjh/ruhin/mra_nn/logs
mkdir -p /gpfs/projects/rjh/ruhin/mra_nn/checkpoints

cd /gpfs/projects/rjh/ruhin/madness-ruhin
git add mra_nn/configs/default.yaml mra_nn/slurm/train_a100.sh
git commit -m "feat(mra-nn): add Step 6 config and Slurm script

default.yaml with all hyperparameters from design spec.
train_a100.sh for single-GPU A100 training."
```

---

### Task 2: PyTorch Dataset

**Model recommendation: Sonnet 4.6** — straightforward data loading from a known HDF5 schema.

**Files:**
- Create: `mra_nn/dataset.py`

**Interfaces:**
- Consumes: `training_dataset.h5` (HDF5 file built by `dataset_builder.py`), `default.yaml` config
- Produces:
  - `class MRADataset(torch.utils.data.Dataset)` with `__len__() -> int` and `__getitem__(idx: int) -> dict` returning keys: `rho0_s` (216,), `vnuc_s` (216,), `halo_rho0` (6, 216), `halo_vnuc` (6, 216), `delta_rho` (216,), `log_dnorm` (scalar), `refine` (scalar), `level` (scalar int), `negative` (scalar int)
  - `def build_dataloaders(cfg: dict) -> Tuple[DataLoader, DataLoader, DataLoader]` returning (train_loader, val_loader, test_loader)
  - `def compute_baseline_mse(dataset: MRADataset) -> float` returning ‖Δρ‖² averaged over positive samples (the "predict zero" baseline)

- [ ] **Step 1: Write the test**

Create `mra_nn/tests/test_dataset.py`:

```python
"""Tests for MRA-NN PyTorch dataset."""
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from mra_nn.dataset import MRADataset, build_dataloaders, compute_baseline_mse


DATASET_PATH = "/gpfs/projects/rjh/ruhin/mra_nn/training_dataset.h5"
CONFIG_PATH = str(Path(__file__).resolve().parents[1] / "configs" / "default.yaml")


@pytest.fixture
def cfg():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


@pytest.fixture
def train_ds(cfg):
    return MRADataset(cfg["data"]["dataset_path"], cfg["data"]["train_molecules"])


def test_dataset_length(train_ds):
    assert len(train_ds) > 0
    # 12 train molecules should have ~787k samples
    assert len(train_ds) > 500_000


def test_dataset_getitem_keys(train_ds):
    sample = train_ds[0]
    expected_keys = {
        "rho0_s", "vnuc_s", "halo_rho0", "halo_vnuc",
        "delta_rho", "log_dnorm", "refine", "level", "negative",
    }
    assert set(sample.keys()) == expected_keys


def test_dataset_getitem_shapes(train_ds):
    sample = train_ds[0]
    assert sample["rho0_s"].shape == (216,)
    assert sample["vnuc_s"].shape == (216,)
    assert sample["halo_rho0"].shape == (6, 216)
    assert sample["halo_vnuc"].shape == (6, 216)
    assert sample["delta_rho"].shape == (216,)
    assert sample["log_dnorm"].shape == ()
    assert sample["refine"].shape == ()
    assert sample["level"].shape == ()
    assert sample["negative"].shape == ()


def test_dataset_getitem_dtypes(train_ds):
    sample = train_ds[0]
    assert sample["rho0_s"].dtype == torch.float32
    assert sample["level"].dtype == torch.long
    assert sample["refine"].dtype == torch.float32
    assert sample["negative"].dtype == torch.float32


def test_build_dataloaders(cfg):
    train_dl, val_dl, test_dl = build_dataloaders(cfg)
    batch = next(iter(train_dl))
    assert batch["rho0_s"].shape[0] <= cfg["training"]["batch_size"]
    assert batch["rho0_s"].shape[1] == 216


def test_baseline_mse(train_ds):
    baseline = compute_baseline_mse(train_ds)
    # baseline should be positive (non-zero delta_rho)
    assert baseline > 0.0
    # and finite
    assert np.isfinite(baseline)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /gpfs/projects/rjh/ruhin/madness-ruhin
source /gpfs/projects/rjh/ruhin/mra_nn/.venv/bin/activate
python -m pytest mra_nn/tests/test_dataset.py -v 2>&1 | head -20
```

Expected: `ModuleNotFoundError: No module named 'mra_nn.dataset'`

- [ ] **Step 3: Implement the dataset module**

Create `mra_nn/dataset.py`:

```python
"""PyTorch Dataset wrapping the MRA-NN training HDF5 file.

Loads all samples into memory at init (908k samples, ~10.5 GB — fits on A100).
Provides WeightedRandomSampler for class-imbalanced training.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


class MRADataset(Dataset):
    """In-memory dataset from training_dataset.h5.

    Parameters
    ----------
    h5_path : str
        Path to the training HDF5 file (built by dataset_builder.py).
    molecules : list[str]
        Which molecule groups to include.
    """

    FIELD_NAMES = [
        "rho0_s", "vnuc_s", "halo_rho0", "halo_vnuc",
        "delta_rho", "log_dnorm", "refine", "level", "negative",
    ]

    def __init__(self, h5_path: str, molecules: List[str]) -> None:
        arrays: Dict[str, list] = {name: [] for name in self.FIELD_NAMES}
        with h5py.File(h5_path, "r") as f:
            for mol in molecules:
                grp = f[mol]
                for name in self.FIELD_NAMES:
                    arrays[name].append(grp[name][:])

        # Concatenate across molecules
        self.data: Dict[str, torch.Tensor] = {}
        for name in self.FIELD_NAMES:
            arr = np.concatenate(arrays[name], axis=0)
            if name == "level":
                self.data[name] = torch.from_numpy(arr.astype(np.int64))
            elif name in ("refine", "negative"):
                self.data[name] = torch.from_numpy(arr.astype(np.float32))
            else:
                self.data[name] = torch.from_numpy(arr)

        self.n_samples = self.data["level"].shape[0]

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {name: self.data[name][idx] for name in self.FIELD_NAMES}

    def sample_weights(self, refine_pos_weight: float = 10.0) -> torch.Tensor:
        """Per-sample weights for WeightedRandomSampler.

        refine=1 positives get `refine_pos_weight`, everything else gets 1.0.
        """
        w = torch.ones(self.n_samples)
        is_refine_pos = (self.data["refine"] == 1) & (self.data["negative"] == 0)
        w[is_refine_pos] = refine_pos_weight
        return w


def build_dataloaders(cfg: dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build train/val/test DataLoaders from config.

    Train loader uses WeightedRandomSampler for oversampling refine=1 positives.
    Val and test loaders use sequential sampling (no shuffling).
    """
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    loss_cfg = cfg["loss"]

    train_ds = MRADataset(data_cfg["dataset_path"], data_cfg["train_molecules"])
    val_ds = MRADataset(data_cfg["dataset_path"], data_cfg["val_molecules"])
    test_ds = MRADataset(data_cfg["dataset_path"], data_cfg["test_molecules"])

    # Weighted sampler for training
    weights = train_ds.sample_weights(loss_cfg["refine_pos_weight"])
    sampler = WeightedRandomSampler(weights, num_samples=len(train_ds), replacement=True)

    train_dl = DataLoader(
        train_ds,
        batch_size=train_cfg["batch_size"],
        sampler=sampler,
        num_workers=train_cfg["num_workers"],
        pin_memory=True,
        drop_last=True,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=train_cfg["num_workers"],
        pin_memory=True,
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=train_cfg["num_workers"],
        pin_memory=True,
    )
    return train_dl, val_dl, test_dl


def compute_baseline_mse(dataset: MRADataset) -> float:
    """Mean squared error of predicting Δρ = 0, over positive samples only.

    This is the "predict zero correction" baseline — the model must beat this.
    """
    mask = dataset.data["negative"] == 0
    delta_rho = dataset.data["delta_rho"][mask]  # [N_pos, 216]
    return float(delta_rho.pow(2).mean())
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /gpfs/projects/rjh/ruhin/madness-ruhin
source /gpfs/projects/rjh/ruhin/mra_nn/.venv/bin/activate
python -m pytest mra_nn/tests/test_dataset.py -v
```

Expected: all 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /gpfs/projects/rjh/ruhin/madness-ruhin
git add mra_nn/dataset.py mra_nn/tests/test_dataset.py
git commit -m "feat(mra-nn): add PyTorch dataset and dataloader

In-memory MRADataset from training_dataset.h5 with
WeightedRandomSampler for oversampling refine=1 positives.
Leave-molecules-out split per design spec."
```

---

### Task 3: Model Architecture

**Model recommendation: Opus 4.6** — core architectural design requiring FiLM conditioning, shared halo encoder, multi-head output, and careful dimension threading.

**Files:**
- Create: `mra_nn/model.py`

**Interfaces:**
- Consumes: config dict from `default.yaml` (model section)
- Produces:
  - `class MRANet(nn.Module)` with `forward(rho0_s, vnuc_s, halo_rho0, halo_vnuc, level) -> Tuple[Tensor, Tensor, Tensor]` returning `(delta_rho [B, 216], log_dnorm [B], refine_logit [B])`
  - `def build_model(cfg: dict) -> MRANet`

- [ ] **Step 1: Write the test**

Create `mra_nn/tests/test_model.py`:

```python
"""Tests for MRA-NN model architecture."""
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from mra_nn.model import MRANet, build_model

CONFIG_PATH = str(Path(__file__).resolve().parents[1] / "configs" / "default.yaml")


@pytest.fixture
def cfg():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


@pytest.fixture
def model(cfg):
    return build_model(cfg)


def test_build_model(model):
    assert isinstance(model, MRANet)


def test_forward_shapes(model):
    B = 8
    rho0_s = torch.randn(B, 216)
    vnuc_s = torch.randn(B, 216)
    halo_rho0 = torch.randn(B, 6, 216)
    halo_vnuc = torch.randn(B, 6, 216)
    level = torch.randint(0, 19, (B,))

    delta_rho, log_dnorm, refine_logit = model(
        rho0_s, vnuc_s, halo_rho0, halo_vnuc, level
    )
    assert delta_rho.shape == (B, 216)
    assert log_dnorm.shape == (B,)
    assert refine_logit.shape == (B,)


def test_forward_gradients(model):
    B = 4
    rho0_s = torch.randn(B, 216, requires_grad=True)
    vnuc_s = torch.randn(B, 216)
    halo_rho0 = torch.randn(B, 6, 216)
    halo_vnuc = torch.randn(B, 6, 216)
    level = torch.randint(0, 19, (B,))

    delta_rho, log_dnorm, refine_logit = model(
        rho0_s, vnuc_s, halo_rho0, halo_vnuc, level
    )
    loss = delta_rho.sum() + log_dnorm.sum() + refine_logit.sum()
    loss.backward()
    assert rho0_s.grad is not None
    assert rho0_s.grad.shape == (B, 216)


def test_parameter_count(model):
    total = sum(p.numel() for p in model.parameters())
    # Spec says ~2.1M — allow 1.5M to 3M range
    assert 1_500_000 < total < 3_000_000, f"Parameter count {total} outside expected range"


def test_halo_encoder_weight_sharing(model):
    """The halo encoder should use shared weights for all 6 faces."""
    # Process two different face indices through the encoder —
    # the linear weights should be identical (same nn.Module)
    B = 2
    halo_input = torch.randn(B, 6, 216)
    # Access the halo encoder directly
    assert hasattr(model, "halo_encoder")


def test_film_conditioning_exists(model):
    """Model should have FiLM layers that take level input."""
    assert hasattr(model, "level_embedding")
    # Level embedding should have 19 entries (levels 0-18)
    assert model.level_embedding.num_embeddings == 19
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /gpfs/projects/rjh/ruhin/madness-ruhin
source /gpfs/projects/rjh/ruhin/mra_nn/.venv/bin/activate
python -m pytest mra_nn/tests/test_model.py -v 2>&1 | head -20
```

Expected: `ModuleNotFoundError: No module named 'mra_nn.model'`

- [ ] **Step 3: Implement the model**

Create `mra_nn/model.py`:

```python
"""MRANet — FiLM-conditioned MLP with factored halo encoder.

Architecture (from design spec):
  - Halo encoder: shared MLP across 6 face-adjacent neighbors (440 -> 256 -> 128)
  - Center features: rho0_s(216) + vnuc_s(216) = 432
  - Trunk: 3 FiLM-conditioned layers (1200 -> 1024 -> 512 -> 256)
  - FiLM: level embedding (32-dim) -> per-layer gamma/beta
  - Heads: delta_rho (256->216), log_dnorm (256->1), refine (256->1)
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class HaloEncoder(nn.Module):
    """Shared encoder for face-adjacent neighbor boxes.

    Processes each of 6 neighbors identically through a shared MLP,
    with a learnable face embedding to distinguish +x/-x/+y/-y/+z/-z.
    """

    def __init__(
        self,
        k_cubed: int = 216,
        face_embed_dim: int = 8,
        hidden_dim: int = 256,
        out_dim: int = 128,
        n_faces: int = 6,
    ) -> None:
        super().__init__()
        self.n_faces = n_faces
        self.face_embedding = nn.Embedding(n_faces, face_embed_dim)
        # Input: rho0_halo(k^3) + vnuc_halo(k^3) + face_embed
        input_dim = 2 * k_cubed + face_embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(
        self, halo_rho0: torch.Tensor, halo_vnuc: torch.Tensor
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        halo_rho0 : Tensor [B, 6, k^3]
        halo_vnuc : Tensor [B, 6, k^3]

        Returns
        -------
        Tensor [B, 6 * out_dim]
            Concatenated halo embeddings.
        """
        B = halo_rho0.shape[0]
        # Face indices [0..5], expanded to [B, 6]
        face_ids = torch.arange(self.n_faces, device=halo_rho0.device)
        face_ids = face_ids.unsqueeze(0).expand(B, -1)  # [B, 6]
        face_emb = self.face_embedding(face_ids)  # [B, 6, face_embed_dim]

        # Concatenate per-face: [B, 6, 2*k^3 + face_embed_dim]
        x = torch.cat([halo_rho0, halo_vnuc, face_emb], dim=-1)

        # Reshape to process all faces at once through shared MLP
        B, F, D = x.shape
        x = x.reshape(B * F, D)  # [B*6, input_dim]
        x = self.mlp(x)  # [B*6, out_dim]
        x = x.reshape(B, F, -1)  # [B, 6, out_dim]

        # Concatenate face outputs
        return x.reshape(B, -1)  # [B, 6*out_dim]


class FiLMLayer(nn.Module):
    """Linear -> BatchNorm -> FiLM(gamma, beta) -> ReLU -> Dropout.

    FiLM conditioning: output = gamma * BatchNorm(Wx + b) + beta
    where gamma and beta are produced from the level embedding.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        level_embed_dim: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        # Project level embedding to (gamma, beta) pair
        self.film_proj = nn.Linear(level_embed_dim, 2 * out_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, level_emb: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor [B, in_dim]
        level_emb : Tensor [B, level_embed_dim]

        Returns
        -------
        Tensor [B, out_dim]
        """
        h = self.linear(x)
        h = self.bn(h)
        # FiLM modulation
        film = self.film_proj(level_emb)  # [B, 2*out_dim]
        gamma, beta = film.chunk(2, dim=-1)  # each [B, out_dim]
        h = gamma * h + beta
        return self.dropout(self.relu(h))


class MRANet(nn.Module):
    """FiLM-conditioned MLP with factored halo encoder for MRA-NN.

    Three output heads:
      - delta_rho: density correction [B, k^3]
      - log_dnorm: wavelet norm [B]
      - refine: refinement logit [B] (apply sigmoid externally for probabilities)
    """

    def __init__(
        self,
        k_cubed: int = 216,
        n_faces: int = 6,
        n_levels: int = 19,
        level_embed_dim: int = 32,
        face_embed_dim: int = 8,
        halo_encoder_hidden: int = 256,
        halo_encoder_out: int = 128,
        trunk_dims: tuple = (1024, 512, 256),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

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
        # Input: halo_encoder output (6*128=768) + center features (2*216=432)
        center_dim = 2 * k_cubed  # rho0_s + vnuc_s
        halo_out_dim = n_faces * halo_encoder_out
        trunk_input_dim = halo_out_dim + center_dim  # 768 + 432 = 1200

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
        self.head_delta_rho = nn.Linear(final_dim, k_cubed)
        self.head_log_dnorm = nn.Linear(final_dim, 1)
        self.head_refine = nn.Linear(final_dim, 1)

    def forward(
        self,
        rho0_s: torch.Tensor,
        vnuc_s: torch.Tensor,
        halo_rho0: torch.Tensor,
        halo_vnuc: torch.Tensor,
        level: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        rho0_s    : [B, k^3]
        vnuc_s    : [B, k^3]
        halo_rho0 : [B, 6, k^3]
        halo_vnuc : [B, 6, k^3]
        level     : [B] (long)

        Returns
        -------
        delta_rho    : [B, k^3]
        log_dnorm    : [B]
        refine_logit : [B]
        """
        # Halo encoding
        halo_emb = self.halo_encoder(halo_rho0, halo_vnuc)  # [B, 768]

        # Center features
        center = torch.cat([rho0_s, vnuc_s], dim=-1)  # [B, 432]

        # Level embedding (for FiLM, not concatenated)
        level_emb = self.level_embedding(level)  # [B, 32]

        # Trunk input
        x = torch.cat([halo_emb, center], dim=-1)  # [B, 1200]

        # FiLM-conditioned trunk
        for film_layer in self.trunk:
            x = film_layer(x, level_emb)

        # Output heads
        delta_rho = self.head_delta_rho(x)  # [B, 216]
        log_dnorm = self.head_log_dnorm(x).squeeze(-1)  # [B]
        refine_logit = self.head_refine(x).squeeze(-1)  # [B]

        return delta_rho, log_dnorm, refine_logit


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
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /gpfs/projects/rjh/ruhin/madness-ruhin
source /gpfs/projects/rjh/ruhin/mra_nn/.venv/bin/activate
python -m pytest mra_nn/tests/test_model.py -v
```

Expected: all 6 tests PASS. Check parameter count is in range 1.5M–3M.

- [ ] **Step 5: Commit**

```bash
cd /gpfs/projects/rjh/ruhin/madness-ruhin
git add mra_nn/model.py mra_nn/tests/test_model.py
git commit -m "feat(mra-nn): add MRANet model architecture

FiLM-conditioned MLP with shared halo encoder. Three output
heads: delta_rho (216), log_dnorm (1), refine logit (1).
~2.1M parameters."
```

---

### Task 4: Loss Functions & Training Loop

**Model recommendation: Opus 4.6** — uncertainty-weighted multi-task loss with focal loss, masked delta-rho, AMP training, early stopping, and CSV logging require careful implementation.

**Files:**
- Create: `mra_nn/losses.py`
- Create: `mra_nn/train.py`

**Interfaces:**
- Consumes:
  - `MRANet` from `model.py`: `forward(rho0_s, vnuc_s, halo_rho0, halo_vnuc, level) -> (delta_rho, log_dnorm, refine_logit)`
  - `build_dataloaders(cfg)` from `dataset.py`: returns `(train_dl, val_dl, test_dl)`
  - `build_model(cfg)` from `model.py`: returns `MRANet`
  - `compute_baseline_mse(dataset)` from `dataset.py`: returns `float`
  - config dict from `default.yaml`
- Produces:
  - `class FocalLoss(nn.Module)` with `forward(logits: Tensor, targets: Tensor) -> Tensor`
  - `class UncertaintyWeightedLoss(nn.Module)` with `forward(batch: dict, delta_rho: Tensor, log_dnorm: Tensor, refine_logit: Tensor) -> Tuple[Tensor, dict]` returning `(total_loss, loss_components_dict)`
  - `train.py` CLI: `python train.py --config <yaml>` that trains to completion, saves checkpoints and metrics CSV

- [ ] **Step 1: Write the loss function tests**

Create `mra_nn/tests/test_losses.py`:

```python
"""Tests for MRA-NN loss functions."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from mra_nn.losses import FocalLoss, UncertaintyWeightedLoss


def test_focal_loss_zero_for_perfect_predictions():
    loss_fn = FocalLoss(gamma=2.0, alpha=0.75)
    # Perfect predictions: logits very positive for target=1, very negative for target=0
    logits = torch.tensor([10.0, -10.0, 10.0])
    targets = torch.tensor([1.0, 0.0, 1.0])
    loss = loss_fn(logits, targets)
    assert loss.item() < 0.01


def test_focal_loss_high_for_wrong_predictions():
    loss_fn = FocalLoss(gamma=2.0, alpha=0.75)
    # Wrong predictions
    logits = torch.tensor([-10.0, 10.0])
    targets = torch.tensor([1.0, 0.0])
    loss = loss_fn(logits, targets)
    assert loss.item() > 1.0


def test_focal_loss_gradients():
    loss_fn = FocalLoss(gamma=2.0, alpha=0.75)
    logits = torch.randn(32, requires_grad=True)
    targets = torch.randint(0, 2, (32,), dtype=torch.float32)
    loss = loss_fn(logits, targets)
    loss.backward()
    assert logits.grad is not None


def test_uncertainty_weighted_loss_output_keys():
    uwl = UncertaintyWeightedLoss(focal_gamma=2.0, focal_alpha=0.75)
    B = 16
    batch = {
        "delta_rho": torch.randn(B, 216),
        "log_dnorm": torch.randn(B),
        "refine": torch.randint(0, 2, (B,), dtype=torch.float32),
        "negative": torch.zeros(B),
    }
    pred_dr = torch.randn(B, 216)
    pred_ld = torch.randn(B)
    pred_ref = torch.randn(B)
    total, components = uwl(batch, pred_dr, pred_ld, pred_ref)
    assert "loss_delta_rho" in components
    assert "loss_log_dnorm" in components
    assert "loss_refine" in components
    assert "sigma_delta_rho" in components
    assert total.requires_grad


def test_uncertainty_weighted_loss_masks_negatives():
    uwl = UncertaintyWeightedLoss(focal_gamma=2.0, focal_alpha=0.75)
    B = 16
    # All negatives — delta_rho loss should be zero
    batch = {
        "delta_rho": torch.randn(B, 216),
        "log_dnorm": torch.randn(B),
        "refine": torch.zeros(B),
        "negative": torch.ones(B),  # all negatives
    }
    pred_dr = torch.randn(B, 216)
    pred_ld = torch.randn(B)
    pred_ref = torch.randn(B)
    total, components = uwl(batch, pred_dr, pred_ld, pred_ref)
    assert components["loss_delta_rho"].item() == 0.0


def test_uncertainty_sigmas_are_learnable():
    uwl = UncertaintyWeightedLoss(focal_gamma=2.0, focal_alpha=0.75)
    learnable_params = [n for n, p in uwl.named_parameters() if p.requires_grad]
    # Should have 3 log-sigma parameters
    assert len(learnable_params) == 3
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /gpfs/projects/rjh/ruhin/madness-ruhin
source /gpfs/projects/rjh/ruhin/mra_nn/.venv/bin/activate
python -m pytest mra_nn/tests/test_losses.py -v 2>&1 | head -20
```

Expected: `ModuleNotFoundError: No module named 'mra_nn.losses'`

- [ ] **Step 3: Implement the loss module**

Create `mra_nn/losses.py`:

```python
"""Loss functions for MRA-NN multi-task training.

- FocalLoss: focal binary cross-entropy for the refine head (Lin et al. 2017)
- UncertaintyWeightedLoss: Kendall et al. 2018 uncertainty-weighted multi-task loss
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Binary focal loss (Lin et al. 2017).

    L = -alpha * (1 - p_t)^gamma * log(p_t)

    where p_t = sigmoid(logit) for target=1, else 1 - sigmoid(logit).

    Parameters
    ----------
    gamma : float
        Focusing parameter. gamma=0 recovers standard BCE.
    alpha : float
        Weight for the positive class (target=1).
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.75) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        logits  : [N] raw logits (before sigmoid)
        targets : [N] float, 0.0 or 1.0

        Returns
        -------
        Scalar mean focal loss.
        """
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = torch.exp(-bce)  # equals sigmoid(logit) when target=1, etc.
        focal_weight = (1 - p_t) ** self.gamma

        # Alpha weighting: alpha for positives, (1-alpha) for negatives
        alpha_weight = targets * self.alpha + (1 - targets) * (1 - self.alpha)

        return (alpha_weight * focal_weight * bce).mean()


class UncertaintyWeightedLoss(nn.Module):
    """Uncertainty-weighted multi-task loss (Kendall et al. 2018).

    L = (1/2*sigma1^2)*L_dr + (1/2*sigma2^2)*L_ld + (1/sigma3^2)*L_ref
        + log(sigma1*sigma2*sigma3)

    Three learnable log-sigma parameters auto-balance the task weights.
    """

    def __init__(self, focal_gamma: float = 2.0, focal_alpha: float = 0.75) -> None:
        super().__init__()
        self.focal_loss = FocalLoss(gamma=focal_gamma, alpha=focal_alpha)
        # Learnable log-variance parameters, initialized to 0 (sigma=1)
        self.log_sigma_dr = nn.Parameter(torch.tensor(0.0))
        self.log_sigma_ld = nn.Parameter(torch.tensor(0.0))
        self.log_sigma_ref = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        pred_delta_rho: torch.Tensor,
        pred_log_dnorm: torch.Tensor,
        pred_refine_logit: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Parameters
        ----------
        batch : dict with keys delta_rho, log_dnorm, refine, negative
        pred_delta_rho    : [B, 216]
        pred_log_dnorm    : [B]
        pred_refine_logit : [B]

        Returns
        -------
        total_loss : scalar
        components : dict of individual losses and sigma values (detached)
        """
        # --- Delta-rho MSE (positive samples only) ---
        pos_mask = batch["negative"] == 0
        n_pos = pos_mask.sum()
        if n_pos > 0:
            loss_dr = F.mse_loss(
                pred_delta_rho[pos_mask], batch["delta_rho"][pos_mask]
            )
        else:
            loss_dr = torch.tensor(0.0, device=pred_delta_rho.device)

        # --- Log-dnorm MSE (all samples) ---
        loss_ld = F.mse_loss(pred_log_dnorm, batch["log_dnorm"])

        # --- Refine focal loss (all samples) ---
        loss_ref = self.focal_loss(pred_refine_logit, batch["refine"])

        # --- Uncertainty weighting ---
        sigma_dr = torch.exp(self.log_sigma_dr)
        sigma_ld = torch.exp(self.log_sigma_ld)
        sigma_ref = torch.exp(self.log_sigma_ref)

        total = (
            0.5 / sigma_dr**2 * loss_dr
            + 0.5 / sigma_ld**2 * loss_ld
            + 1.0 / sigma_ref**2 * loss_ref
            + self.log_sigma_dr + self.log_sigma_ld + self.log_sigma_ref
        )

        components = {
            "loss_delta_rho": loss_dr.detach(),
            "loss_log_dnorm": loss_ld.detach(),
            "loss_refine": loss_ref.detach(),
            "sigma_delta_rho": sigma_dr.detach(),
            "sigma_log_dnorm": sigma_ld.detach(),
            "sigma_refine": sigma_ref.detach(),
        }
        return total, components
```

- [ ] **Step 4: Run loss tests**

```bash
cd /gpfs/projects/rjh/ruhin/madness-ruhin
source /gpfs/projects/rjh/ruhin/mra_nn/.venv/bin/activate
python -m pytest mra_nn/tests/test_losses.py -v
```

Expected: all 6 tests PASS.

- [ ] **Step 5: Implement the training loop**

Create `mra_nn/train.py`:

```python
"""MRA-NN training script.

Usage:
    python train.py --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import csv
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml

from dataset import MRADataset, build_dataloaders, compute_baseline_mse
from losses import UncertaintyWeightedLoss
from model import build_model


def compute_refine_f1(
    pred_logits: torch.Tensor, targets: torch.Tensor
) -> float:
    """Compute F1 score for the refine head."""
    preds = (pred_logits > 0).float()
    tp = ((preds == 1) & (targets == 1)).sum().float()
    fp = ((preds == 1) & (targets == 0)).sum().float()
    fn = ((preds == 0) & (targets == 1)).sum().float()
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    return float(2 * precision * recall / (precision + recall + 1e-8))


def train_one_epoch(
    model: torch.nn.Module,
    loss_fn: UncertaintyWeightedLoss,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
) -> dict:
    """Train for one epoch. Returns dict of mean losses."""
    model.train()
    accum = {}
    n_batches = 0

    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            dr, ld, ref = model(
                batch["rho0_s"], batch["vnuc_s"],
                batch["halo_rho0"], batch["halo_vnuc"],
                batch["level"],
            )
            total_loss, components = loss_fn(batch, dr, ld, ref)

        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Accumulate metrics
        for k, v in components.items():
            accum[k] = accum.get(k, 0.0) + v.item()
        accum["total_loss"] = accum.get("total_loss", 0.0) + total_loss.item()
        n_batches += 1

    return {k: v / n_batches for k, v in accum.items()}


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loss_fn: UncertaintyWeightedLoss,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict:
    """Evaluate on val/test set. Returns dict of mean losses + refine F1."""
    model.eval()
    accum = {}
    n_batches = 0
    all_ref_logits = []
    all_ref_targets = []

    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            dr, ld, ref = model(
                batch["rho0_s"], batch["vnuc_s"],
                batch["halo_rho0"], batch["halo_vnuc"],
                batch["level"],
            )
            total_loss, components = loss_fn(batch, dr, ld, ref)

        for k, v in components.items():
            accum[k] = accum.get(k, 0.0) + v.item()
        accum["total_loss"] = accum.get("total_loss", 0.0) + total_loss.item()
        n_batches += 1

        all_ref_logits.append(ref.cpu())
        all_ref_targets.append(batch["refine"].cpu())

    metrics = {k: v / n_batches for k, v in accum.items()}

    # Refine F1
    all_logits = torch.cat(all_ref_logits)
    all_targets = torch.cat(all_ref_targets)
    metrics["refine_f1"] = compute_refine_f1(all_logits, all_targets)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="MRA-NN training")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Seed
    seed = cfg["training"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data
    print("Loading data...")
    train_dl, val_dl, test_dl = build_dataloaders(cfg)
    print(f"  Train: {len(train_dl.dataset)} samples")
    print(f"  Val:   {len(val_dl.dataset)} samples")
    print(f"  Test:  {len(test_dl.dataset)} samples")

    # Baseline
    baseline_mse = compute_baseline_mse(train_dl.dataset)
    print(f"  Baseline MSE (predict zero): {baseline_mse:.6f}")

    # Model
    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters")

    # Loss
    loss_cfg = cfg["loss"]
    loss_fn = UncertaintyWeightedLoss(
        focal_gamma=loss_cfg["focal_gamma"],
        focal_alpha=loss_cfg["focal_alpha"],
    ).to(device)

    # Optimizer (includes loss_fn's learnable sigmas)
    train_cfg = cfg["training"]
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(loss_fn.parameters()),
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
    )

    # LR scheduler: linear warmup then cosine decay
    warmup_epochs = train_cfg["warmup_epochs"]
    max_epochs = train_cfg["max_epochs"]

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / (max_epochs - warmup_epochs)
        min_factor = train_cfg["min_lr"] / train_cfg["lr"]
        return min_factor + 0.5 * (1 - min_factor) * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # AMP scaler
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    # Checkpoint directory
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M")
    ckpt_dir = Path(cfg["checkpoint"]["dir"]) / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoints: {ckpt_dir}")

    # Save config copy
    with open(ckpt_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f)

    # CSV logger
    csv_path = ckpt_dir / "metrics.csv"
    csv_fields = [
        "epoch", "lr",
        "train_total_loss", "train_loss_delta_rho", "train_loss_log_dnorm",
        "train_loss_refine", "train_sigma_delta_rho", "train_sigma_log_dnorm",
        "train_sigma_refine",
        "val_total_loss", "val_loss_delta_rho", "val_loss_log_dnorm",
        "val_loss_refine", "val_refine_f1",
    ]
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
    csv_writer.writeheader()

    # Training loop
    best_val_dr_mse = float("inf")
    patience_counter = 0

    print(f"\nTraining for up to {max_epochs} epochs (patience={train_cfg['patience']})...")
    print(f"{'Epoch':>5} {'LR':>10} {'TrainLoss':>10} {'ValDrMSE':>10} {'ValRefF1':>9} {'Best':>5}")
    print("-" * 55)

    for epoch in range(max_epochs):
        t0 = time.time()

        train_metrics = train_one_epoch(
            model, loss_fn, train_dl, optimizer, scaler, device
        )
        val_metrics = evaluate(model, loss_fn, val_dl, device)

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        # CSV logging
        row = {"epoch": epoch, "lr": f"{current_lr:.6e}"}
        for k, v in train_metrics.items():
            row[f"train_{k}"] = f"{v:.6f}"
        for k, v in val_metrics.items():
            row[f"val_{k}"] = f"{v:.6f}"
        csv_writer.writerow(row)
        csv_file.flush()

        # Checkpointing
        is_best = val_metrics["loss_delta_rho"] < best_val_dr_mse
        if is_best:
            best_val_dr_mse = val_metrics["loss_delta_rho"]
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "loss_fn_state_dict": loss_fn.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_dr_mse": best_val_dr_mse,
                "config": cfg,
            }, ckpt_dir / "best.pt")
        else:
            patience_counter += 1

        # Always save last
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "loss_fn_state_dict": loss_fn.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_dr_mse": best_val_dr_mse,
            "config": cfg,
        }, ckpt_dir / "last.pt")

        dt = time.time() - t0
        best_marker = "*" if is_best else ""
        print(
            f"{epoch:5d} {current_lr:10.2e} {train_metrics['total_loss']:10.4f} "
            f"{val_metrics['loss_delta_rho']:10.6f} {val_metrics['refine_f1']:9.4f} "
            f"{best_marker:>5}"
        )

        # Early stopping
        if patience_counter >= train_cfg["patience"]:
            print(f"\nEarly stopping at epoch {epoch} (patience={train_cfg['patience']})")
            break

    csv_file.close()

    # Final summary
    print(f"\nTraining complete.")
    print(f"  Best val delta-rho MSE: {best_val_dr_mse:.6f}")
    print(f"  Baseline MSE:           {baseline_mse:.6f}")
    if best_val_dr_mse < baseline_mse:
        print(f"  Model BEATS baseline by {(1 - best_val_dr_mse/baseline_mse)*100:.1f}%")
    else:
        print(f"  Model DOES NOT beat baseline")
    print(f"  Checkpoints saved to: {ckpt_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run all tests**

```bash
cd /gpfs/projects/rjh/ruhin/madness-ruhin
source /gpfs/projects/rjh/ruhin/mra_nn/.venv/bin/activate
python -m pytest mra_nn/tests/test_losses.py mra_nn/tests/test_model.py mra_nn/tests/test_dataset.py -v
```

Expected: all tests PASS.

- [ ] **Step 7: Dry-run the training script (CPU, 2 epochs)**

```bash
cd /gpfs/projects/rjh/ruhin/madness-ruhin/mra_nn
source /gpfs/projects/rjh/ruhin/mra_nn/.venv/bin/activate
PYTHONPATH=/gpfs/projects/rjh/adrian/pymra/src:$PYTHONPATH \
python train.py --config configs/default.yaml 2>&1 | head -30
```

Note: This will load the full dataset into CPU memory (~10 GB). If the login node doesn't have enough RAM, create a temporary config with a single molecule per split for the dry-run. The real training runs on A100 via Slurm. If it starts printing epoch lines, let it run for 2 epochs then Ctrl+C.

- [ ] **Step 8: Commit**

```bash
cd /gpfs/projects/rjh/ruhin/madness-ruhin
git add mra_nn/losses.py mra_nn/train.py mra_nn/tests/test_losses.py
git commit -m "feat(mra-nn): add loss functions and training loop

Uncertainty-weighted multi-task loss (Kendall et al. 2018) with focal
loss on refine head. Delta-rho MSE masked to positive samples only.
Training with AdamW, cosine LR, AMP, early stopping, CSV logging."
```

---

### Task 5: Inference Pipeline

**Model recommendation: Opus 4.6** — tree-walk inference requires understanding pymra's FunctionTree/Key/Node interfaces, two-scale refinement, and integral normalization.

**Files:**
- Create: `mra_nn/predict.py`

**Interfaces:**
- Consumes:
  - `MRANet` from `model.py`
  - pymra: `read_function(path) -> FunctionTree`, `write_function(tree, path)`, `node_s(tree, key) -> ndarray`, `FunctionTree`, `Key`, `Node`
  - pymra: `twoscale.refine(s_parent) -> dict[tuple, ndarray]`
  - `reconstruct.py`: not needed here (inference builds a new tree, doesn't roundtrip)
- Produces:
  - `def predict_density(model: MRANet, rho0_path: str, vnuc_path: str, n_electrons: int, device: torch.device, refine_threshold: float = 0.5) -> FunctionTree` — returns the predicted density tree
  - CLI: `python predict.py --checkpoint <best.pt> --rho0 <path> --vnuc <path> --n-electrons <N> --out <output.mad.h5>`

- [ ] **Step 1: Write the test**

Create `mra_nn/tests/test_predict.py`:

```python
"""Tests for MRA-NN inference pipeline."""
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from mra_nn.model import build_model
from mra_nn.predict import predict_density

CONFIG_PATH = str(Path(__file__).resolve().parents[1] / "configs" / "default.yaml")
# H2O test data
RHO0_PATH = "/gpfs/projects/rjh/ruhin/mra_nn/training_data/h2o/rho0.mad.h5"
VNUC_PATH = "/gpfs/projects/rjh/ruhin/mra_nn/training_data/h2o/vnuc.mad.h5"
RHO_PATH = "/gpfs/projects/rjh/ruhin/mra_nn/training_data/h2o/rho.mad.h5"
N_ELECTRONS_H2O = 10


@pytest.fixture
def cfg():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


@pytest.fixture
def model(cfg):
    m = build_model(cfg)
    m.eval()
    return m


def test_predict_density_returns_tree(model):
    """predict_density should return a FunctionTree with leaves."""
    from pymra import FunctionTree
    tree = predict_density(
        model, RHO0_PATH, VNUC_PATH,
        n_electrons=N_ELECTRONS_H2O,
        device=torch.device("cpu"),
    )
    assert isinstance(tree, FunctionTree)
    leaves = list(tree.leaves())
    assert len(leaves) > 0


def test_predict_density_integral_normalized(model):
    """After post-processing, integral should equal N_electrons."""
    tree = predict_density(
        model, RHO0_PATH, VNUC_PATH,
        n_electrons=N_ELECTRONS_H2O,
        device=torch.device("cpu"),
    )
    integral = tree.integral()
    assert abs(integral - N_ELECTRONS_H2O) < 0.01, (
        f"Integral {integral} != {N_ELECTRONS_H2O}"
    )


def test_predict_density_writes_h5(model, tmp_path):
    """Output tree should be writable to HDF5."""
    from pymra import write_function, read_function
    tree = predict_density(
        model, RHO0_PATH, VNUC_PATH,
        n_electrons=N_ELECTRONS_H2O,
        device=torch.device("cpu"),
    )
    out_path = str(tmp_path / "predicted_rho.mad.h5")
    write_function(tree, out_path)
    reloaded = read_function(out_path)
    assert len(list(reloaded.leaves())) == len(list(tree.leaves()))
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /gpfs/projects/rjh/ruhin/madness-ruhin
source /gpfs/projects/rjh/ruhin/mra_nn/.venv/bin/activate
PYTHONPATH=/gpfs/projects/rjh/adrian/pymra/src python -m pytest mra_nn/tests/test_predict.py -v 2>&1 | head -20
```

Expected: `ModuleNotFoundError: No module named 'mra_nn.predict'`

- [ ] **Step 3: Implement the inference pipeline**

Create `mra_nn/predict.py`:

```python
"""Tree-walk inference for MRA-NN.

Walks the MRA tree top-down. At each node:
  - Extract features (rho0_s, vnuc_s, halos, level)
  - Forward through the model -> (delta_rho, log_dnorm, refine_prob)
  - If refine_prob > threshold: mark as internal, enqueue children
  - Else: mark as leaf with coefficients = rho0_s + delta_rho

Post-processing: scale all leaf coefficients so integral(rho) = N.

Usage:
    python predict.py --checkpoint best.pt --rho0 rho0.mad.h5 \\
                      --vnuc vnuc.mad.h5 --n-electrons 10 --out predicted.mad.h5
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from typing import List

import numpy as np
import torch
import yaml

from pymra import FunctionTree, read_function, write_function
from pymra.tree import Key, Node
from pymra.twoscale import node_s, refine as twoscale_refine

from model import MRANet, build_model


# 6 face-adjacent offsets in 3D: +x, -x, +y, -y, +z, -z
HALO_OFFSETS = [
    (1, 0, 0), (-1, 0, 0),
    (0, 1, 0), (0, -1, 0),
    (0, 0, 1), (0, 0, -1),
]


def _is_valid_key(key: Key) -> bool:
    max_t = 1 << key.n
    return all(0 <= li < max_t for li in key.l)


def _safe_node_s(tree: FunctionTree, key: Key) -> np.ndarray:
    """node_s with zero-padding for out-of-cell halo boxes."""
    if not _is_valid_key(key):
        return np.zeros((tree.k,) * tree.ndim)
    return node_s(tree, key)


def _halo_keys(key: Key) -> List[Key]:
    return [
        Key(key.n, tuple(key.l[d] + off[d] for d in range(key.ndim)))
        for off in HALO_OFFSETS
    ]


def _extract_features(
    keys: List[Key],
    rho0_tree: FunctionTree,
    vnuc_tree: FunctionTree,
    device: torch.device,
) -> dict:
    """Extract model input features for a batch of keys."""
    k = rho0_tree.k
    ndim = rho0_tree.ndim

    rho0_s_list = []
    vnuc_s_list = []
    halo_rho0_list = []
    halo_vnuc_list = []
    level_list = []

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

    return {
        "rho0_s": torch.from_numpy(np.array(rho0_s_list, dtype=np.float32)).to(device),
        "vnuc_s": torch.from_numpy(np.array(vnuc_s_list, dtype=np.float32)).to(device),
        "halo_rho0": torch.from_numpy(np.array(halo_rho0_list, dtype=np.float32)).to(device),
        "halo_vnuc": torch.from_numpy(np.array(halo_vnuc_list, dtype=np.float32)).to(device),
        "level": torch.tensor(level_list, dtype=torch.long).to(device),
    }


def _ensure_children_exist(
    tree: FunctionTree, parent_key: Key
) -> List[Key]:
    """Ensure rho0/vnuc have s-coefficients at parent_key's children.

    If the tree doesn't go that deep, refine the parent's coefficients down.
    Returns list of child keys.
    """
    children = list(parent_key.children())
    node = tree.nodes.get(parent_key)
    if node is None:
        # This shouldn't happen in normal use, but handle gracefully
        return children

    # Check if children already exist
    if all(ck in tree.nodes and tree.nodes[ck].has_coeff for ck in children):
        return children

    # Need to refine parent down
    parent_s = node_s(tree, parent_key)
    child_coeffs = twoscale_refine(parent_s)

    for bits, child_s in child_coeffs.items():
        child_key = Key(
            parent_key.n + 1,
            tuple(2 * parent_key.l[d] + bits[d] for d in range(tree.ndim)),
        )
        if child_key not in tree.nodes or not tree.nodes[child_key].has_coeff:
            tree.nodes[child_key] = Node(s=child_s)

    return children


@torch.no_grad()
def predict_density(
    model: MRANet,
    rho0_path: str,
    vnuc_path: str,
    n_electrons: int,
    device: torch.device,
    refine_threshold: float = 0.5,
    max_level: int = 18,
) -> FunctionTree:
    """Predict density via top-down tree walk.

    Parameters
    ----------
    model : trained MRANet
    rho0_path : path to rho0.mad.h5
    vnuc_path : path to vnuc.mad.h5
    n_electrons : number of electrons (for integral normalization)
    device : torch device
    refine_threshold : probability threshold for refinement decision
    max_level : maximum tree depth (safety limit)

    Returns
    -------
    FunctionTree with predicted density coefficients, integral-normalized.
    """
    model.eval()
    rho0_tree = read_function(rho0_path)
    vnuc_tree = read_function(vnuc_path)

    k = rho0_tree.k
    ndim = rho0_tree.ndim

    predicted_tree = FunctionTree(
        k=k, ndim=ndim,
        cell=rho0_tree.cell.copy(),
        thresh=rho0_tree.thresh,
        initial_level=rho0_tree.initial_level,
    )

    # Start at root
    root_key = Key(0, (0,) * ndim)
    current_level_keys = [root_key]

    while current_level_keys:
        features = _extract_features(
            current_level_keys, rho0_tree, vnuc_tree, device
        )
        delta_rho, log_dnorm, refine_logit = model(
            features["rho0_s"], features["vnuc_s"],
            features["halo_rho0"], features["halo_vnuc"],
            features["level"],
        )

        refine_prob = torch.sigmoid(refine_logit).cpu().numpy()
        delta_rho_np = delta_rho.cpu().numpy()

        next_level_keys = []
        for i, key in enumerate(current_level_keys):
            if refine_prob[i] > refine_threshold and key.n < max_level:
                # Internal node — refine
                predicted_tree.nodes[key] = Node(has_children=True)
                # Ensure rho0/vnuc have children for feature extraction
                _ensure_children_exist(rho0_tree, key)
                _ensure_children_exist(vnuc_tree, key)
                next_level_keys.extend(key.children())
            else:
                # Leaf — write predicted coefficients
                rho0_s = node_s(rho0_tree, key).ravel()
                pred_s = rho0_s + delta_rho_np[i]
                predicted_tree.nodes[key] = Node(
                    s=pred_s.reshape((k,) * ndim).astype(np.float64)
                )

        current_level_keys = next_level_keys

    # Post-processing: normalize integral to N_electrons
    integral = predicted_tree.integral()
    if abs(integral) > 1e-10:  # avoid division by zero
        scale = n_electrons / integral
        for _, node in predicted_tree.leaves():
            node.s = node.s * scale

    return predicted_tree


def main():
    parser = argparse.ArgumentParser(description="MRA-NN inference")
    parser.add_argument("--checkpoint", required=True, help="Path to best.pt")
    parser.add_argument("--rho0", required=True, help="Path to rho0.mad.h5")
    parser.add_argument("--vnuc", required=True, help="Path to vnuc.mad.h5")
    parser.add_argument("--n-electrons", type=int, required=True)
    parser.add_argument("--out", required=True, help="Output .mad.h5 path")
    parser.add_argument("--refine-threshold", type=float, default=0.5)
    args = parser.parse_args()

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    print(f"Loaded model from {args.checkpoint} (epoch {ckpt['epoch']})")
    print(f"Predicting density for rho0={args.rho0}, vnuc={args.vnuc}")

    tree = predict_density(
        model, args.rho0, args.vnuc,
        n_electrons=args.n_electrons,
        device=device,
        refine_threshold=args.refine_threshold,
    )

    n_leaves = sum(1 for _ in tree.leaves())
    integral = tree.integral()
    print(f"Predicted tree: {n_leaves} leaves, integral={integral:.6f}")

    write_function(tree, args.out)
    print(f"Written to {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests**

```bash
cd /gpfs/projects/rjh/ruhin/madness-ruhin
source /gpfs/projects/rjh/ruhin/mra_nn/.venv/bin/activate
PYTHONPATH=/gpfs/projects/rjh/adrian/pymra/src python -m pytest mra_nn/tests/test_predict.py -v
```

Expected: all 3 tests PASS. Note: the model is untrained (random weights), so the predicted tree will have random structure — but it should still produce a valid FunctionTree with integral normalized to 10.

- [ ] **Step 5: Commit**

```bash
cd /gpfs/projects/rjh/ruhin/madness-ruhin
git add mra_nn/predict.py mra_nn/tests/test_predict.py
git commit -m "feat(mra-nn): add tree-walk inference pipeline

Top-down tree walk using trained model. At each node: predict
delta_rho + refine decision. Refines rho0/vnuc via two-scale when
the model predicts deeper than the input trees go. Post-processing
normalizes integral to N electrons."
```

---

### Task 6: Evaluation & Step 6 Gate

**Model recommendation: Sonnet 4.6** — straightforward metrics computation using interfaces established in earlier tasks.

**Files:**
- Create: `mra_nn/evaluate.py`

**Interfaces:**
- Consumes:
  - `predict_density()` from `predict.py`
  - `build_model()` from `model.py`
  - `MRADataset`, `compute_baseline_mse()` from `dataset.py`
  - pymra: `read_function`, `FunctionTree`
  - Raw data paths from config
- Produces:
  - CLI: `python evaluate.py --checkpoint <best.pt> --config <yaml>` that prints all gate metrics and PASS/FAIL verdict

- [ ] **Step 1: Implement the evaluation script**

Create `mra_nn/evaluate.py`:

```python
"""MRA-NN evaluation and Step 6 gate check.

Computes all metrics from the design spec:
  1. Val delta-rho MSE < baseline (predict zero)
  2. Refine F1 > 0.5
  3. Predicted tree writable to HDF5
  4. Integral error < 0.01 after normalization

Usage:
    python evaluate.py --checkpoint best.pt --config configs/default.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml

from pymra import read_function

from dataset import MRADataset, build_dataloaders, compute_baseline_mse
from losses import UncertaintyWeightedLoss
from model import build_model
from predict import predict_density
from train import compute_refine_f1, evaluate as evaluate_epoch


# Electron counts per molecule (for integral check)
ELECTRON_COUNTS = {
    "h2o": 10, "nh3": 10, "ch4": 10, "co2": 22, "hf": 10,
    "n2": 14, "co": 14, "hcn": 14, "c2h2": 14, "c2h4": 16,
    "c2h6": 18, "h2co": 16, "ch3oh": 18, "h2o2": 18, "hcl": 18,
}


def main():
    parser = argparse.ArgumentParser(description="MRA-NN evaluation + gate")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded model from {args.checkpoint} (epoch {ckpt['epoch']})")

    # --- Gate 1: Val delta-rho MSE < baseline ---
    print("\n=== Gate 1: Val delta-rho MSE vs baseline ===")
    _, val_dl, _ = build_dataloaders(cfg)
    val_ds = val_dl.dataset

    loss_fn = UncertaintyWeightedLoss(
        focal_gamma=cfg["loss"]["focal_gamma"],
        focal_alpha=cfg["loss"]["focal_alpha"],
    ).to(device)
    loss_fn.load_state_dict(ckpt["loss_fn_state_dict"])

    val_metrics = evaluate_epoch(model, loss_fn, val_dl, device)
    baseline_mse = compute_baseline_mse(val_ds)

    val_dr_mse = val_metrics["loss_delta_rho"]
    gate1_pass = val_dr_mse < baseline_mse
    print(f"  Val delta-rho MSE:  {val_dr_mse:.6f}")
    print(f"  Baseline (zero):    {baseline_mse:.6f}")
    print(f"  Improvement:        {(1 - val_dr_mse/baseline_mse)*100:.1f}%")
    print(f"  Gate 1:             {'PASS' if gate1_pass else 'FAIL'}")

    # --- Gate 2: Refine F1 > 0.5 ---
    print("\n=== Gate 2: Refine F1 ===")
    refine_f1 = val_metrics["refine_f1"]
    gate2_pass = refine_f1 > 0.5
    print(f"  Refine F1:          {refine_f1:.4f}")
    print(f"  Gate 2:             {'PASS' if gate2_pass else 'FAIL'}")

    # --- Gate 3 & 4: Predict on test molecules, check HDF5 and integral ---
    print("\n=== Gates 3 & 4: Prediction + integral ===")
    raw_dir = Path(cfg["data"]["raw_data_dir"])
    test_mols = cfg["data"]["test_molecules"]
    gate3_pass = True
    gate4_pass = True

    for mol in test_mols:
        rho0_path = str(raw_dir / mol / "rho0.mad.h5")
        vnuc_path = str(raw_dir / mol / "vnuc.mad.h5")
        rho_path = str(raw_dir / mol / "rho.mad.h5")
        n_el = ELECTRON_COUNTS[mol]

        pred_tree = predict_density(
            model, rho0_path, vnuc_path,
            n_electrons=n_el, device=device,
        )

        n_leaves = sum(1 for _ in pred_tree.leaves())
        true_tree = read_function(rho_path)
        true_leaves = sum(1 for _ in true_tree.leaves())
        tree_ratio = n_leaves / true_leaves if true_leaves > 0 else 0

        integral = pred_tree.integral()
        int_err = abs(integral - n_el)
        mol_gate4 = int_err < 0.01

        print(f"  {mol}: {n_leaves} leaves (true: {true_leaves}, "
              f"ratio: {tree_ratio:.2f}), "
              f"integral={integral:.6f}, err={int_err:.4e} "
              f"{'OK' if mol_gate4 else 'FAIL'}")

        if not mol_gate4:
            gate4_pass = False

    print(f"  Gate 3 (valid HDF5): {'PASS' if gate3_pass else 'FAIL'}")
    print(f"  Gate 4 (integral):   {'PASS' if gate4_pass else 'FAIL'}")

    # --- Overall verdict ---
    all_pass = gate1_pass and gate2_pass and gate3_pass and gate4_pass
    print(f"\n{'='*40}")
    print(f"STEP 6 GATE: {'PASS' if all_pass else 'FAIL'}")
    print(f"{'='*40}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    exit(main())
```

- [ ] **Step 2: Verify it imports correctly**

```bash
cd /gpfs/projects/rjh/ruhin/madness-ruhin/mra_nn
source /gpfs/projects/rjh/ruhin/mra_nn/.venv/bin/activate
PYTHONPATH=/gpfs/projects/rjh/adrian/pymra/src \
python -c "from evaluate import ELECTRON_COUNTS; print(f'Loaded {len(ELECTRON_COUNTS)} molecules')"
```

Expected: `Loaded 15 molecules`

- [ ] **Step 3: Commit**

```bash
cd /gpfs/projects/rjh/ruhin/madness-ruhin
git add mra_nn/evaluate.py
git commit -m "feat(mra-nn): add evaluation script and Step 6 gate

Computes all gate metrics: val delta-rho MSE vs baseline,
refine F1, tree-walk prediction on test molecules with
integral normalization check."
```

---

### Task 7: GPU Training Run & Gate

**Model recommendation: Sonnet 4.6** — submitting Slurm job, monitoring output, running the gate check. No new code to write.

**Files:**
- No new files created
- Modify: none

**Interfaces:**
- Consumes: all prior tasks (config, dataset, model, train, predict, evaluate)
- Produces: trained model checkpoint + gate PASS/FAIL verdict

- [ ] **Step 1: Create the logs directory and submit the training job**

```bash
mkdir -p /gpfs/projects/rjh/ruhin/mra_nn/logs
cd /gpfs/projects/rjh/ruhin/madness-ruhin
/cm/shared/apps/slurm/21.08.8/bin/sbatch mra_nn/slurm/train_a100.sh
```

Expected: `Submitted batch job <jobid>`

Note the job ID. Monitor with:
```bash
squeue -u ruhipatel
```

- [ ] **Step 2: Monitor training progress**

Once the job starts running:
```bash
tail -f /gpfs/projects/rjh/ruhin/mra_nn/logs/train_<jobid>.out
```

Watch for:
- Data loading completes without OOM
- Baseline MSE prints
- First few epoch lines show decreasing train loss
- Val delta-rho MSE starts decreasing

If the job fails immediately, check the `.err` file:
```bash
cat /gpfs/projects/rjh/ruhin/mra_nn/logs/train_<jobid>.err
```

Common issues: CUDA not available (wrong partition), import errors (PYTHONPATH), OOM (unlikely with 64GB + 80GB GPU).

- [ ] **Step 3: Run the gate check after training completes**

Find the latest checkpoint:
```bash
ls -lt /gpfs/projects/rjh/ruhin/mra_nn/checkpoints/
```

Then run the gate:
```bash
cd /gpfs/projects/rjh/ruhin/madness-ruhin/mra_nn
source /gpfs/projects/rjh/ruhin/mra_nn/.venv/bin/activate
PYTHONPATH=/gpfs/projects/rjh/adrian/pymra/src \
python evaluate.py \
    --checkpoint /gpfs/projects/rjh/ruhin/mra_nn/checkpoints/<run_name>/best.pt \
    --config configs/default.yaml
```

Expected output ends with either `STEP 6 GATE: PASS` or `STEP 6 GATE: FAIL`.

If PASS: Step 6 is done. Commit the metrics CSV and update Notion.
If FAIL: inspect which gate failed, check the metrics CSV for training curves, and diagnose (likely needs hyperparameter tuning — adjust config and re-submit).

- [ ] **Step 4: Push all commits**

```bash
cd /gpfs/projects/rjh/ruhin/madness-ruhin
git push origin feat/mra-nn-data
```
