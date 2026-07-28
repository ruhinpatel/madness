# MRA-NN Step 6: MLP Model & Training — Design Spec

**Date:** 2026-07-27
**Author:** Ruhi Patel + Claude (Opus 4.6)
**Status:** Approved
**Branch:** `feat/mra-nn-data`

---

## 1. Problem Statement

Train a neural network that predicts converged electron density from the promolecular
density and nuclear potential, both represented in the MRA (multiwavelet) basis. The
predicted density serves as a better SCF initial guess than the promolecular density
(SAD baseline) in MADNESS, reducing convergence iterations.

### Inputs (per box in the MRA tree)

| Feature | Dimension | Description |
|---------|-----------|-------------|
| `rho0_s` | 216 (k^3, k=6) | Promolecular density s-coefficients at the box |
| `vnuc_s` | 216 | Nuclear potential s-coefficients at the box |
| `halo_rho0` | 6 x 216 | rho0 s-coefficients at 6 face-adjacent neighbor boxes |
| `halo_vnuc` | 6 x 216 | vnuc s-coefficients at 6 face-adjacent neighbor boxes |
| `level` | 1 (integer 0-18) | Tree depth of the box |

**Total raw input dimension:** 3,024 (level handled via FiLM conditioning, not concatenated)

### Outputs (per box)

| Target | Dimension | Type | Description |
|--------|-----------|------|-------------|
| `delta_rho` | 216 | Regression | Density correction: rho - rho0 s-coefficients |
| `log_dnorm` | 1 | Regression | log(||d_rho||), wavelet norm; range [-30, 0] |
| `refine` | 1 | Classification | 1 = refine further, 0 = leaf |

### Success Criteria

| Priority | Metric | Target |
|----------|--------|--------|
| Primary | SCF iteration reduction on held-out molecules (vs. SAD) | >= 30% fewer iterations |
| Secondary | Val delta-rho MSE | < ||rho - rho0||^2 (beats "predict zero") |
| Tertiary | Refine F1 score on validation molecule | > 0.7 |

**Note:** SCF iteration measurement requires Step 7 (C++ integration). The Step 6 gate
uses the secondary and tertiary criteria to verify the model has learned useful predictions
before proceeding to integration.

---

## 2. Decision Log

Every design decision is recorded here with alternatives considered and reasoning.

### Decision 1: Inference Mode

**Question:** How does the model get used inside MADNESS at inference time?

| Option | Description | Pros | Cons |
|--------|-------------|------|------|
| **(A) Offline batch** | Predict on all boxes of the rho0 tree, write corrected density to HDF5, MADNESS reads it | Simple; no C++ bridge needed | Cannot improve tree structure; refine + log||d|| signals wasted |
| **(B) Online tree-walk** | Walk tree top-down; at each node predict (delta-rho, log||d||, refine); descend if refine=1 | Exploits all 3 outputs; model decides tree shape | Sequential inference; needs C++/Python bridge eventually |

**Selected: (B) Online tree-walk**

**Reasoning:** The dataset was explicitly designed with refine flags and below-leaf negatives
to teach where the tree should be refined. The log||d|| signal is the key advantage over
Gong et al.'s GED-CRN approach (which needs multi-resolution comparison). Option (A) throws
away both signals. Sequential inference is acceptable — tree-walking ~5k-15k nodes at
~1ms/node is negligible vs. SCF cost. For prototyping, inference runs in Python (pymra);
C++ bridge deferred to Step 7.

---

### Decision 2: Level Handling

**Question:** One model for all tree levels, or separate models per level?

| Option | Description | Pros | Cons |
|--------|-------------|------|------|
| **(A) Level as input feature** | Add level as scalar to input vector | Simplest | Model dominated by levels 11-14 (68% of data); no explicit level adaptation |
| **(B) FiLM conditioning** | Level embedding produces per-layer scale/shift (gamma, beta) | Explicit level-dependent behavior; proven technique | Moderate implementation complexity |
| **(C) Per-level models** | Separate MLP per level (or level band) | Maximum specialization | Levels 0-6 have <= 960 samples each — far too few; need banding anyway |

**Selected: (B) FiLM conditioning**

**Reasoning:** Level distribution is extremely non-uniform (level 0: 15 samples, level 13:
251k samples). Per-level models (C) would starve at extreme levels. A flat input (A) forces
the model to learn level-dependent behavior implicitly from a single scalar. FiLM (B) gives
each trunk layer level-specific affine transforms — the model architecture directly encodes
"behavior should vary by level" without fragmenting the data.

---

### Decision 3: Loss Function Strategy

**Question:** How to combine the three task losses?

| Option | Description | Pros | Cons |
|--------|-------------|------|------|
| **(A) Manual weighted sum** | L = lambda1*MSE + lambda2*MSE + lambda3*BCE | Simple | Brittle; sensitive to relative scales; requires manual tuning |
| **(B) Uncertainty-weighted** (Kendall et al. 2018) | Learnable log-variance per task; model learns optimal weighting | Automatic; proven; 3 extra scalar parameters | Slightly more complex loss computation |
| **(C) GradNorm** | Dynamically balance gradient magnitudes across tasks | Most sophisticated balancing | Extra hyperparameter (asymmetry alpha); training complexity |

**Selected: (B) Uncertainty-weighted**

**Reasoning:** Three tasks with different scales (216-dim MSE vs. scalar MSE vs. binary focal
loss) make manual weighting fragile. Uncertainty weighting is nearly free (3 learnable scalars),
well-understood, and eliminates a hyperparameter search dimension. GradNorm is overkill for
3 tasks.

**Sub-decision — delta-rho loss scope:**
- Delta-rho MSE computed on **positive samples only** (negative samples masked out)
- Negatives exist to train the refine head ("don't refine here"), not the density predictor
- log||d|| and refine losses computed on all samples

**Sub-decision — refine head loss:**
- Focal loss (gamma=2, alpha=0.75) instead of weighted BCE
- Focal loss naturally down-weights confident easy negatives rather than just scaling magnitude
- gamma=2, alpha=0.75 are standard starting values from Lin et al. 2017

---

### Decision 4: Input Architecture

**Question:** How should the network ingest the structured input (center + 6 halos)?

| Option | Description | Pros | Cons |
|--------|-------------|------|------|
| **(A) Flat MLP** | Concatenate all 3,025 dims, feed through layers | Trivial implementation | First layer alone = ~6M params (3025x2048); no structural inductive bias |
| **(B) Factored halo encoder** | Shared MLP processes each halo, concat embeddings with center features | Encodes neighbor symmetry; fewer parameters; better data efficiency | Moderate implementation |

**Selected: (B) Factored halo encoder**

**Reasoning:** The 6 halos are structurally equivalent — a neighbor's coefficients should be
processed the same way regardless of which face it's on (modulo face identity). With 908k
samples, reducing the parameter count of the first layer from ~6M to ~200k via weight sharing
is significant for generalization. Face identity is captured by a small learnable embedding.

---

### Decision 5: Data Splitting

**Question:** How to split 15 molecules for train/val/test?

| Option | Description | Pros | Cons |
|--------|-------------|------|------|
| **(A) Random sample split** | Shuffle all 908k, split 80/10/10 | Maximum training data | Same-molecule leakage between splits; dishonest evaluation |
| **(B) Leave-molecules-out** | Hold out entire molecules | Honest generalization; matches SCF eval (per-molecule) | Fewer effective training points |
| **(C) K-fold over molecules** | Rotate held-out molecules | Confidence intervals; paper-ready | 5x compute |

**Selected: (B) for development, (C) for final reporting**

**Reasoning:** The headline metric (SCF iteration reduction) is per-molecule, so evaluation
must be per-molecule. Random splitting would let the model memorize molecule-specific patterns.
K-fold is deferred to final reporting to save compute during prototyping.

**Fixed development split:**

| Set | Molecules | Rationale |
|-----|-----------|-----------|
| Train | h2o, nh3, ch4, co2, hf, n2, co, hcn, c2h4, c2h6, h2co, hcl | 12 molecules, bulk of data |
| Val | ch3oh | Mid-size organic, ~60k samples |
| Test | h2o2, c2h2 | Different chemistry — peroxide + triple bond |

---

### Decision 6: Integral Constraint

**Question:** How to enforce integral(rho) = N (total electrons)?

| Option | Description | Pros | Cons |
|--------|-------------|------|------|
| **(A) No constraint** | Train and hope | Simple | No guarantee |
| **(B) Soft penalty** | Add lambda*(integral - N)^2 to loss | Approximate enforcement during training | Molecule-aware batching required; may distort local predictions |
| **(C) Post-processing normalization** | Scale all delta-rho by N / integral(rho_predicted) after inference | Exact; zero training complexity | Uniform scaling — doesn't fix regional imbalances |

**Selected: (C) Post-processing normalization**

**Reasoning:** Exact integral enforcement with zero training complexity. The scaling is
physically reasonable — if the integral is off by 0.1%, scaling all coefficients by 0.999
barely changes local shape. Upgrade to (B) only if post-processing scaling proves too coarse
(systematic regional over/under-prediction).

---

### Decision 7: Negative Sample Handling

**Question:** How to use the negative (below-leaf) samples during training?

| Option | Description | Pros | Cons |
|--------|-------------|------|------|
| **(i) Positives only** | Train on 113k positive samples; ignore negatives | Simpler dataloader | Refine head never sees "don't refine" examples |
| **(ii) All samples, mask delta-rho** | All 908k samples; delta-rho loss masked on negatives; refine/log||d|| loss on everything | Refine head learns from negatives | Larger batches needed; imbalanced sampling |

**Selected: (ii) All samples with masked delta-rho loss**

**Reasoning:** The refine head must see negative examples to learn "don't refine here."
Oversampling handles class imbalance: refine=1 positives weighted 10x in the sampler.

---

## 3. Architecture

### 3.1 Overview

```
Input:  rho0_s(216) + vnuc_s(216) + 6x[halo_rho0(216) + halo_vnuc(216)] + level
                                    |
                    ┌───────────────┼───────────────┐
                    v               v               v
            Halo Encoder      Center Block     Level Embedding
           (shared, x6)        (432-dim)        (32-dim)
            -> 6x128             |                 |
            concat=768           |          FiLM γ,β at each layer
                    |            |               |
                    └──────> Trunk MLP <─────────┘
                         1200 -> 1024 -> 512 -> 256
                              (3 FiLM layers)
                                    |
                    ┌───────────────┼───────────────┐
                    v               v               v
                delta-rho       log||d||         refine
               (256->216)      (256->1)         (256->1)
                linear          linear        linear+sigmoid
```

### 3.2 Halo Encoder (shared weights)

Processes each of the 6 face-adjacent neighbor boxes identically:

- **Input per neighbor:** rho0_halo_i(216) + vnuc_halo_i(216) + face_embedding_i(8) = 440
- **Face embedding:** 6 learnable 8-dim vectors (one per face: +x, -x, +y, -y, +z, -z)
- **Architecture:** Linear(440, 256) -> ReLU -> Linear(256, 128)
- **Aggregation:** Concatenate 6 outputs -> 768-dim vector
- **Weight sharing:** Same encoder parameters for all 6 neighbors; face identity captured by embedding

### 3.3 Level Conditioning (FiLM)

Feature-wise Linear Modulation applied at each trunk layer:

- **Level embedding:** Lookup table, 19 entries (levels 0-18), 32-dim each
- **Per trunk layer:** Linear(32, 2*hidden_dim) -> split into (gamma, beta)
- **Application:** `output = gamma * BatchNorm(linear(input)) + beta`

This gives each level its own affine transform of each hidden representation, allowing
the model to behave differently at different tree depths without separate models.

### 3.4 Trunk MLP

Three FiLM-conditioned layers:

| Layer | Input | Output | Activation | Dropout |
|-------|-------|--------|------------|---------|
| 1 | 1,200 (768 halo + 432 center) | 1,024 | ReLU | 0.1 |
| 2 | 1,024 | 512 | ReLU | 0.1 |
| 3 | 512 | 256 | ReLU | 0.1 |

The level embedding (32-dim) does NOT concatenate into the trunk input. Instead, it feeds
the FiLM conditioning at each layer (producing gamma/beta affine parameters). This keeps
level information as a modulation signal rather than a direct feature.

Each layer: `x -> Linear -> FiLM(BatchNorm, gamma, beta) -> ReLU -> Dropout`

### 3.5 Output Heads

| Head | Architecture | Activation | Loss target |
|------|-------------|------------|-------------|
| delta-rho | Linear(256, 216) | None (linear) | MSE on positives only |
| log\|\|d\|\| | Linear(256, 1) | None (linear) | MSE on all samples |
| refine | Linear(256, 1) | Sigmoid | Focal loss on all samples |

### 3.6 Parameter Count

| Component | Parameters |
|-----------|-----------|
| Halo encoder | ~140k |
| Level embeddings | ~600 |
| FiLM projections (3 layers) | ~120k |
| Trunk MLP | ~1.8M |
| Output heads | ~56k |
| Uncertainty weights | 3 |
| **Total** | **~2.1M** |

Deliberately small — 908k samples cannot support a significantly larger model without overfitting.

---

## 4. Training Configuration

### 4.1 Loss Function

```
L = (1 / 2*sigma_1^2) * L_delta_rho
  + (1 / 2*sigma_2^2) * L_log_dnorm
  + (1 / sigma_3^2)   * L_refine
  + log(sigma_1 * sigma_2 * sigma_3)
```

- `sigma_1, sigma_2, sigma_3`: learnable (initialized as log_sigma = 0)
- `L_delta_rho`: MSE, positive samples only, averaged over 216 dimensions
- `L_log_dnorm`: MSE, all samples
- `L_refine`: Focal loss, gamma=2, alpha=0.75, all samples

### 4.2 Optimizer & Schedule

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning rate | 1e-3 |
| Weight decay | 1e-4 |
| LR schedule | Cosine decay to 1e-5 |
| Warmup | 5 epochs, linear |
| Max epochs | 200 |
| Early stopping | Patience 20, monitoring val delta-rho MSE |
| Precision | Mixed (torch.cuda.amp) |
| Batch size | 4,096 |

### 4.3 Data Loading & Sampling

- All 908k samples loaded into memory at initialization (~10.5 GB, fits on A100)
- WeightedRandomSampler with per-sample weights:
  - refine=1 positive: weight 10.0
  - refine=0 positive: weight 1.0
  - negative: weight 1.0
- Delta-rho loss masked to positive samples within each batch
- DataLoader with num_workers=4, pin_memory=True

### 4.4 Data Split

| Set | Molecules | Count | Samples |
|-----|-----------|-------|---------|
| Train | h2o, nh3, ch4, co2, hf, n2, co, hcn, c2h4, c2h6, h2co, hcl | 12 | ~787k |
| Val | ch3oh | 1 | ~61k |
| Test | h2o2, c2h2 | 2 | ~108k |

Leave-molecules-out for development. K-fold (5-fold, 3 molecules held out per fold) for
final reporting.

---

## 5. Inference Pipeline

### 5.1 Tree-Walk Algorithm

```
function predict_density(model, rho0_tree, vnuc_tree):
    predicted_tree = empty FunctionTree
    queue = [root_key]

    while queue is not empty:
        batch = all keys in queue at current level
        features = extract_features(batch, rho0_tree, vnuc_tree)
        delta_rho, log_d, refine_prob = model.forward(features)

        next_queue = []
        for key, dr, rp in zip(batch, delta_rho, refine_prob):
            if rp > REFINE_THRESHOLD:      # default 0.5, tunable
                predicted_tree.add_internal(key)
                for child in key.children():
                    # refine rho0/vnuc down if needed
                    next_queue.append(child)
            else:
                rho0_s = node_s(rho0_tree, key)
                predicted_tree.add_leaf(key, s = rho0_s + dr)

        queue = next_queue

    # Post-processing: enforce integral = N
    integral = predicted_tree.integral()
    scale = N_electrons / integral
    for leaf in predicted_tree.leaves():
        leaf.s *= scale

    return predicted_tree
```

### 5.2 Refinement Details

When the model predicts refine=1 at a box but rho0/vnuc don't have children there:
- Use `pymra.twoscale.refine` to push parent s-coefficients down to 2^ndim children
- This is exact (two-scale relation) — no approximation involved

### 5.3 Refine Threshold Tuning

The 0.5 default for refine_prob can be tuned post-training:
- Lower threshold -> larger trees (more refined, more accurate, slower)
- Higher threshold -> smaller trees (less refined, less accurate, faster)
- Tune on validation molecule to match tree size ratio ~1.0 vs. true rho tree

---

## 6. Evaluation Protocol

### 6.1 Step 6 Gate (model quality)

All must pass before proceeding to Step 7:

1. Training completes without error on 12 train molecules
2. Val delta-rho MSE < ||rho - rho0||^2 on ch3oh (beats "predict zero correction")
3. Refine F1 > 0.5 on validation molecule
4. Predicted tree for test molecules produces valid HDF5 loadable by pymra
5. Integral error < 0.01 electrons after post-processing normalization

### 6.2 Full Evaluation (with Step 7)

| Metric | Method | Target |
|--------|--------|--------|
| SCF iterations | MADNESS moldft with predicted vs. SAD initial guess | >= 30% reduction |
| delta-rho MSE | Test molecule leaf coefficients | < ||rho - rho0||^2 |
| Refine F1 | Test molecule predictions | > 0.7 |
| Integral error (pre-norm) | \|integral(rho_predicted) - N\| | < 0.1 electrons |
| Tree size ratio | #predicted leaves / #true leaves | 0.8 - 1.2 |

---

## 7. Environment & Infrastructure

### 7.1 Dependencies

Install into existing venv (`/gpfs/projects/rjh/ruhin/mra_nn/.venv/`):
- `torch` (with CUDA support — pip install picks up A100 drivers)
- `pyyaml` (for config files)
- No TensorBoard/W&B — use CSV logging

### 7.2 Compute

| Resource | Spec |
|----------|------|
| Partition | `a100-long` (2-day limit) |
| GPUs | 1x A100 (80 GB) |
| CPUs | 8 cores |
| Memory | 64 GB |
| Estimated training time | < 2 hours for 200 epochs |

Single GPU — 2.1M parameters and in-memory data do not benefit from multi-GPU.

### 7.3 Checkpointing & Logging

- Checkpoints: `/gpfs/projects/rjh/ruhin/mra_nn/checkpoints/<run_name>/`
- Run name: `YYYY-MM-DD_HH-MM` (auto-generated)
- Saved every epoch: `best.pt` (best val delta-rho MSE) + `last.pt`
- Metrics: `checkpoints/<run_name>/metrics.csv` — one row per epoch with all losses, refine F1, LR

### 7.4 File Layout & Claude Model Recommendations

Each file is tagged with the minimum Claude model sufficient for implementation.
Use as reference when switching models during implementation.

```
mra_nn/
  model.py          # MRANet architecture                    → Opus 4.6
  losses.py         # Focal loss + uncertainty-weighted loss  → Opus 4.6
  dataset.py        # PyTorch Dataset wrapping HDF5           → Sonnet 4.6
  train.py          # Training loop, logging, checkpointing   → Opus 4.6
  predict.py        # Tree-walk inference + normalization      → Opus 4.6
  evaluate.py       # Metrics + Step 6 gate                   → Sonnet 4.6
  configs/
    default.yaml    # All hyperparameters                     → Sonnet 4.6
  slurm/
    train_a100.sh   # Slurm submission script                 → Sonnet 4.6
  tests/
    test_dataset.py  # Dataset tests                          → Sonnet 4.6
    test_model.py    # Architecture tests                     → Sonnet 4.6
    test_losses.py   # Loss function tests                    → Sonnet 4.6
    test_predict.py  # Inference tests                        → Sonnet 4.6
```

**Rationale:** Opus 4.6 for files requiring architectural judgment (FiLM conditioning,
multi-task loss balancing, tree-walk inference with pymra interfaces). Sonnet 4.6 for
boilerplate tasks (config, data loading, metrics, Slurm, tests).

### 7.5 Slurm Script

```bash
#!/bin/bash
#SBATCH --partition=a100-long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --job-name=mra-nn-train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

export PATH="/cm/shared/apps/slurm/21.08.8/bin:$PATH"
source /gpfs/projects/rjh/ruhin/mra_nn/.venv/bin/activate
export PYTHONPATH=/gpfs/projects/rjh/adrian/pymra/src:$PYTHONPATH

python /gpfs/projects/rjh/ruhin/madness-ruhin/mra_nn/train.py \
    --config /gpfs/projects/rjh/ruhin/madness-ruhin/mra_nn/configs/default.yaml
```

---

## 8. Dataset Summary

For reference, the training dataset characteristics that shaped this design:

| Property | Value |
|----------|-------|
| Total samples | 908,551 |
| Positive (in-tree) | 113,567 (12.5%) |
| Negative (below-leaf) | 794,984 (87.5%) |
| Refine=1 | 14,194 (1.56% total, 12.5% of positives) |
| Input dimension | 3,024 (216 rho0 + 216 vnuc + 6*216 halo_rho0 + 6*216 halo_vnuc); level via FiLM |
| k | 6 |
| k^3 | 216 |
| Molecules | 15 |
| Peak levels | 11-14 (68.9% of samples) |
| Sparse levels | 0-6 (<= 960 samples each) |
| On-disk size | ~10.5 GB (fits in A100 memory) |

---

## 9. Open Questions (Deferred)

These are recorded for future consideration but explicitly out of scope for Step 6:

1. **Parent/sibling context as features** — Adding parent node coefficients or sibling
   predictions as input could improve accuracy. Deferred because it adds sequential
   dependencies within a level (sibling) or requires two-pass inference (parent context
   at child prediction time). Start simple, add if needed.

2. **One model vs. ensemble** — An ensemble of 3-5 models with different random seeds
   could improve robustness and provide uncertainty estimates. Deferred to after single-model
   baseline is established.

3. **k=8 training data** — Current data is k=6 (prototype). Adrian confirmed k=8 for final
   training. Regenerating data at k=8 changes k^3 from 216 to 512, roughly doubling input
   size. Architecture scales naturally (just change dimensions) but training cost increases.

4. **Graph neural network architecture** — The MRA tree is a graph; GNNs could capture
   multi-hop context. Deferred as significant complexity increase over MLP baseline.

5. **GPU inference inside MADNESS** — Step 7 decision. CPU inference via Python subprocess
   or C++ LibTorch is simpler; GPU adds latency from data transfer but faster for large trees.
