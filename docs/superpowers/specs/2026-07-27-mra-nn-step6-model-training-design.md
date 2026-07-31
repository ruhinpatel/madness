# MRA-NN Step 6: MLP Model & Training — Design Spec

**Date:** 2026-07-27 (updated 2026-07-30)
**Author:** Ruhi Patel + Claude (Opus 4.6)
**Status:** Approved (v2 — k=8, direct coefficients, redundant form)
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
| `rho0_s` | 512 (k^3, k=8) | Promolecular density s-coefficients at the box |
| `vnuc_s` | 512 | Nuclear potential s-coefficients at the box |
| `halo_rho0` | 6 x 512 | rho0 s-coefficients at 6 face-adjacent neighbor boxes |
| `halo_vnuc` | 6 x 512 | vnuc s-coefficients at 6 face-adjacent neighbor boxes |
| `level` | 1 (integer 0-18) | Tree depth of the box |

**Total raw input dimension:** 6,656 (level handled via FiLM conditioning, not concatenated)

### Outputs (per box)

| Target | Dimension | Type | Description |
|--------|-----------|------|-------------|
| `rho_s` | 512 | Regression | Predicted converged density s-coefficients (direct, not delta) |
| `log_dnorm` | 1 | Regression | log(||d||), wavelet norm from redundant form; range [-30, 0] |
| `refine` | 1 | Classification | 1 = refine further, 0 = leaf |

### Success Criteria

| Priority | Metric | Target |
|----------|--------|--------|
| Primary | SCF iteration reduction on held-out molecules (vs. SAD) | >= 30% fewer iterations |
| Secondary | Val rho_s MSE | < ||rho0_s||² (beats using rho0 directly as the guess) |
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
| **(B) Online tree-walk** | Walk tree top-down; at each node predict (rho_s, log||d||, refine); descend if refine=1 | Exploits all 3 outputs; model decides tree shape | Sequential inference; needs C++/Python bridge eventually |

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

**Reasoning:** Three tasks with different scales (512-dim MSE vs. scalar MSE vs. binary focal
loss) make manual weighting fragile. Uncertainty weighting is nearly free (3 learnable scalars),
well-understood, and eliminates a hyperparameter search dimension. GradNorm is overkill for
3 tasks.

**Sub-decision — rho_s loss scope (updated 2026-07-30):**
- rho_s MSE computed on **all samples** — in the redundant form, every node has
  well-defined s-coefficients, so the model should learn accurate coefficients everywhere.
- log||d|| and refine losses also on all samples.

**Sub-decision — refine head loss:**
- Focal loss (gamma=2, alpha=0.75) instead of weighted BCE
- Focal loss naturally down-weights confident easy negatives rather than just scaling magnitude
- gamma=2, alpha=0.75 are standard starting values from Lin et al. 2017

---

### Decision 4: Input Architecture

**Question:** How should the network ingest the structured input (center + 6 halos)?

| Option | Description | Pros | Cons |
|--------|-------------|------|------|
| **(A) Flat MLP** | Concatenate all 6,657 dims, feed through layers | Trivial implementation | First layer alone = ~13M params (6657x2048); no structural inductive bias |
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
| Train | h2o, nh3, ch4, co2, hf, n2, co, hcn, c2h4, c2h6, h2co, hcl, ch3oh | 13 molecules — ch3oh moved here from val (see update below) |
| Val | ch3f | Isoelectronic with training molecules (18 electrons); all bond types C-H, C-F covered; fairer eval than ch3oh |
| Test | h2o2, c2h2 | Different chemistry — peroxide + triple bond |

**Update (2026-07-30):** Original val was ch3oh. Training runs showed the model generalized 9x worse than baseline on ch3oh because the C-O-H alcohol combination is absent from the 12 training molecules. Switched to ch3f, which is chemically closer to the training set (isoelectronic with H₂O/HF/NH₃/CH₄ at 18 electrons; C-H and C-F bonds both present in training). This improved from 9x worse → 2.8x worse. ch3oh was moved to training. ch3f geometry taken from W4-11 dataset at `/gpfs/projects/rjh/ruhin/perf_pipeline/molecules/W4-11/ch3f/struc.xyz`.

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

**Selected: (ii) All samples, all heads**

**Reasoning:** The refine head must see negative examples to learn "don't refine here."
Oversampling handles class imbalance: refine=1 positives weighted 10x in the sampler.

**Updated (2026-07-30):** With the switch from delta prediction to direct coefficients
(Decision 8) and redundant form (Decision 9), the delta-rho masking is no longer needed.
In the redundant form, every node has well-defined s-coefficients (refined down from parents
via two-scale), so rho_s MSE trains on all samples. All three heads now train on all samples.

---

### Decision 8: Target Representation

**Question:** Should the model predict the correction (rho - rho0) or the converged coefficients directly?

| Option | Description | Pros | Cons |
|--------|-------------|------|------|
| **(A) Delta prediction** | Predict rho - rho0 (correction) | Smaller target values; residual learning | Correction ~1800x smaller than signal at k=6; hard to learn |
| **(B) Direct coefficients** | Predict converged rho s-coefficients directly | Larger signal; no cancellation issues; simpler inference | Larger target values; model must learn absolute scale |

**Selected: (B) Direct coefficients**

**Reasoning:** First training run (job 2103119, 2026-07-28, A100) showed delta prediction
failed — the correction is ~1800x smaller than the signal at k=6, and the model learned to
predict zero (30x worse than baseline). Adrian and Robert confirmed: predict coefficients
directly. This also simplifies inference — no `rho0_s + delta_rho` addition step, just use
the predicted s-coefficients as-is.

---

### Decision 9: Redundant Form

**Question:** How to obtain the redundant form (s+d at every node) for the training data?

| Option | Description | Pros | Cons |
|--------|-------------|------|------|
| **(A) C++ dump tool** | Modify `dump_training_functions` to call `make_redundant()` | Single dump step | Requires C++ changes; rebuild; resubmit Slurm |
| **(B) Python via pymra** | Derive s+d at every node using pymra's compress (already validated in Step 2) | No C++ changes; already validated (roundtrip err=1.6e-16) | Extra Python processing step |

**Selected: (B) Python via pymra**

**Reasoning:** pymra compress→reconstruct roundtrip already validated (Step 2 gate, max
err=1.6e-16). No C++ changes needed. Changing the representation = rerun a Python script,
not a MADNESS job — consistent with the core design principle ("all ML-specific logic happens
in Python offline"). Adrian confirmed 2026-07-30.

---

### Decision 10: Training Data Precision

**Question:** What k and thresh to use for training data?

| Option | Description | Pros | Cons |
|--------|-------------|------|------|
| **(A) k=6, thresh=1e-4** | Prototype quality (current data) | Already generated; faster | Below production quality |
| **(B) k=8, thresh=1e-6** | Standard MADNESS production precision | Production-relevant model; standard practice | Must regenerate all data; larger k^3 (512 vs 216) |

**Selected: (B) k=8, thresh=1e-6**

**Reasoning:** Adrian confirmed (2026-07-30) that k=8/thresh=1e-6 is the standard MADNESS
precision for production calculations. k=6 was fine for prototyping but the final model
should train on production-quality data. k^3 goes from 216 to 512, roughly 2.4x the
coefficient dimensions. Architecture scales naturally (just change dimensions).

---

## 3. Architecture

### 3.1 Overview

```
Input:  rho0_s(512) + vnuc_s(512) + 6x[halo_rho0(512) + halo_vnuc(512)] + level
                                    |
                    ┌───────────────┼───────────────┐
                    v               v               v
            Halo Encoder      Center Block     Level Embedding
           (shared, x6)       (1024-dim)        (32-dim)
            -> 6x128             |                 |
            concat=768           |          FiLM γ,β at each layer
                    |            |               |
                    └──────> Trunk MLP <─────────┘
                         1792 -> 1024 -> 512 -> 256
                              (3 FiLM layers)
                                    |
                    ┌───────────────┼───────────────┐
                    v               v               v
                 rho_s          log||d||         refine
               (256->512)      (256->1)         (256->1)
                linear          linear        linear+sigmoid
```

### 3.2 Halo Encoder (shared weights)

Processes each of the 6 face-adjacent neighbor boxes identically:

- **Input per neighbor:** rho0_halo_i(512) + vnuc_halo_i(512) + face_embedding_i(8) = 1032
- **Face embedding:** 6 learnable 8-dim vectors (one per face: +x, -x, +y, -y, +z, -z)
- **Architecture:** Linear(1032, 256) -> ReLU -> Linear(256, 128)
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
| 1 | 1,792 (768 halo + 1024 center) | 1,024 | ReLU | 0.1 |
| 2 | 1,024 | 512 | ReLU | 0.1 |
| 3 | 512 | 256 | ReLU | 0.1 |

The level embedding (32-dim) does NOT concatenate into the trunk input. Instead, it feeds
the FiLM conditioning at each layer (producing gamma/beta affine parameters). This keeps
level information as a modulation signal rather than a direct feature.

Each layer: `x -> Linear -> FiLM(BatchNorm, gamma, beta) -> ReLU -> Dropout`

### 3.5 Output Heads

| Head | Architecture | Activation | Loss target |
|------|-------------|------------|-------------|
| rho_s | Linear(256, 512) | None (linear) | MSE on all samples |
| log\|\|d\|\| | Linear(256, 1) | None (linear) | MSE on all samples |
| refine | Linear(256, 1) | Sigmoid | Focal loss on all samples |

### 3.6 Parameter Count

| Component | Parameters |
|-----------|-----------|
| Halo encoder | ~530k |
| Level embeddings | ~600 |
| FiLM projections (3 layers) | ~120k |
| Trunk MLP | ~2.0M |
| Output heads | ~132k |
| Uncertainty weights | 3 |
| **Total** | **~2.8M** |

Moderately sized — dataset will be larger at k=8 (more tree nodes at higher precision).

---

## 4. Training Configuration

### 4.1 Loss Function

```
L = (1 / 2*sigma_1^2) * L_rho_s
  + (1 / 2*sigma_2^2) * L_log_dnorm
  + (1 / sigma_3^2)   * L_refine
  + log(sigma_1 * sigma_2 * sigma_3)
```

- `sigma_1, sigma_2, sigma_3`: learnable (initialized as log_sigma = 0)
- `L_rho_s`: MSE, all samples, averaged over 512 dimensions
- `L_log_dnorm`: MSE, all samples
- `L_refine`: Focal loss, gamma=2, alpha=0.75, all samples

### 4.2 Optimizer & Schedule

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning rate | 2e-4 |
| Weight decay | 1e-4 |
| LR schedule | Cosine decay to 1e-6 |
| Warmup | 5 epochs, linear |
| Max epochs | 120 (increasing to 240 — model still improving at epoch 119) |
| Early stopping | Patience 20, monitoring **positive-only** val rho_s MSE |
| Precision | Mixed (torch.cuda.amp) |
| Batch size | 4,096 |

**Update (2026-07-30):** LR reduced from 1e-3 → 2e-4 (oscillation observed at epoch 15-30 with 1e-3). min_lr reduced from 1e-5 → 1e-6. Early stopping metric changed from all-sample to positive-only MSE — all-sample saturates to ≈0 by epoch 30 due to 87% negatives and is not informative.

### 4.3 Data Loading & Sampling

- All samples loaded into memory at initialization (fits on A100 80 GB)
- WeightedRandomSampler with per-sample weights:
  - refine=1 positive: weight 10.0
  - refine=0 positive: weight 1.0
  - negative: weight 1.0
- All three heads train on all samples (no masking)
- `pos_rho_weight=10.0` in `UncertaintyWeightedLoss`: in-tree (negative==0) samples weighted 10x in the rho_s MSE to counteract the 87% negative imbalance. Without this, the loss gradient is dominated by negative samples (where rho≈rho0≈0) and the model learns little about the actual density.
- DataLoader with num_workers=4, pin_memory=True

**Update (2026-07-30):** `log_sigma_rs` clamped to ≤0 after each optimizer step (enforces sigma_rs ≤ 1). Without this clamp, the Kendall uncertainty weighting can cause sigma_rs → ∞, zeroing the rho_s gradient (the model learns to "ignore" the hardest task).

### 4.4 Data Split

| Set | Molecules | Count | Samples |
|-----|-----------|-------|---------|
| Train | h2o, nh3, ch4, co2, hf, n2, co, hcn, c2h4, c2h6, h2co, hcl, ch3oh | 13 | ~175,157 |
| Val | ch3f | 1 | TBD (data generation in progress as of 2026-07-30) |
| Test | h2o2, c2h2 | 2 | 32,530 |

**Update (2026-07-30):** ch3oh moved from val → train; ch3f added as val. ch3f dataset generation submitted as Slurm job (gen_ch3f.sh). Sample count for ch3f TBD pending job completion.

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
        rho_s, log_d, refine_prob = model.forward(features)

        next_queue = []
        for key, rs, rp in zip(batch, rho_s, refine_prob):
            if rp > REFINE_THRESHOLD:      # default 0.5, tunable
                predicted_tree.add_internal(key)
                for child in key.children():
                    # refine rho0/vnuc down if needed
                    next_queue.append(child)
            else:
                predicted_tree.add_leaf(key, s = rs)  # direct coefficients

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

1. Training completes without error on 13 train molecules
2. **Positive-only** val rho_s MSE < positive-only rho0 baseline on ch3f (beats using rho0 directly, evaluated on in-tree nodes only — negative/below-leaf samples are excluded because they are synthetic training data with rho≈rho0≈0 and would trivially dominate the metric)
3. Refine F1 > 0.5 on validation molecule (ch3f)
4. Predicted tree for test molecules (h2o2, c2h2) produces valid HDF5 loadable by pymra
5. Integral error < 0.01 electrons after post-processing normalization

**Update (2026-07-30):** Gate 2 changed from all-sample MSE to positive-only MSE. All-sample saturates to ~5e-8 by epoch 30 (dominated by 87% negatives where rho≈rho0≈0) and is not a useful gate. Positive-only baseline for ch3f: 4.846e-7. Current best: 1.375e-6 (2.8x worse). Gates 3-5 pass.

### 6.2 Full Evaluation (with Step 7)

| Metric | Method | Target |
|--------|--------|--------|
| SCF iterations | MADNESS moldft with predicted vs. SAD initial guess | >= 30% reduction |
| rho_s MSE | Test molecule leaf coefficients | < ||rho0_s||² |
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

Single GPU — 2.8M parameters and in-memory data do not benefit from multi-GPU.

### 7.3 Checkpointing & Logging

- Checkpoints: `/gpfs/projects/rjh/ruhin/mra_nn/checkpoints/<run_name>/`
- Run name: `YYYY-MM-DD_HH-MM` (auto-generated)
- Saved every epoch: `best.pt` (best val rho_s MSE) + `last.pt`
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
| Total samples | 207,687 (train 160,364 / val 14,793 / test 32,530) |
| Positive (in-tree) | 25,959 |
| Negative (below-leaf) | 181,728 |
| Refine=1 | 3,243 |
| Input dimension | 6,656 (512 rho0 + 512 vnuc + 6×512 halo_rho0 + 6×512 halo_vnuc); level via FiLM |
| k | 8 |
| k^3 | 512 |
| Molecules | 15 |
| Peak levels | measured from builder output; see per-molecule breakdown below |
| Sparse levels | see per-molecule breakdown below |
| On-disk size | 5.29 GB |

**Per-molecule breakdown (k=8, thresh=1e-6, dataset_builder.py 2026-07-30):**

| Molecule | Split | Total | Positive | Refine=1 | Negative |
|----------|-------|-------|----------|----------|----------|
| c2h2     | test  | 14,409 | 1,801 | 225 | 12,608 |
| c2h4     | train | 15,433 | 1,929 | 241 | 13,504 |
| c2h6     | train | 14,665 | 1,833 | 229 | 12,832 |
| ch3oh    | val   | 14,793 | 1,849 | 231 | 12,944 |
| ch4      | train |  9,545 | 1,193 | 149 |  8,352 |
| co       | train | 11,081 | 1,385 | 173 |  9,696 |
| co2      | train | 17,481 | 2,185 | 273 | 15,296 |
| h2co     | train | 14,409 | 1,801 | 225 | 12,608 |
| h2o      | train | 11,849 | 1,481 | 185 | 10,368 |
| h2o2     | test  | 18,121 | 2,265 | 283 | 15,856 |
| hcl      | train | 18,761 | 2,345 | 293 | 16,416 |
| hcn      | train | 12,617 | 1,577 | 197 | 11,040 |
| hf       | train | 10,313 | 1,289 | 161 |  9,024 |
| n2       | train | 13,385 | 1,673 | 209 | 11,712 |
| nh3      | train | 10,825 | 1,353 | 169 |  9,472 |
| **Total**|       | **207,687** | **25,959** | **3,243** | **181,728** |

Gate check (c2h2, first molecule): leaf coeff max err 2.972e-08 (tol 1e-05) OK; ∫ρ err 2.415e-08 OK — **PASS**

**Note:** Previous k=6/thresh=1e-4 dataset had 908,551 samples (~10.5 GB). k=8/thresh=1e-6
produces 207,687 samples at 5.29 GB — fewer total samples than k=6 because the deeper/finer
trees at k=8 have larger leaf coefficients (512 vs 216 per node) but the MRA grid is more
compact in node count for the same molecule at higher precision.

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

3. ~~**k=8 training data**~~ — **Resolved → Decision 10 (2026-07-30).** k=8/thresh=1e-6 confirmed
   as standard MADNESS production precision. Data regeneration required before retraining.

4. **Graph neural network architecture** — The MRA tree is a graph; GNNs could capture
   multi-hop context. Deferred as significant complexity increase over MLP baseline.

5. **GPU inference inside MADNESS** — Step 7 decision. CPU inference via Python subprocess
   or C++ LibTorch is simpler; GPU adds latency from data transfer but faster for large trees.

---

## 10. Training Observations (2026-07-30)

Seven training runs on the A100 cluster revealed six independent bugs and two data distribution issues. Recorded here to inform future training decisions.

### Bugs Found and Fixed

| # | Bug | Symptom | Fix |
|---|-----|---------|-----|
| 1 | No residual connection | Gate 1: 51% worse than baseline. Head output drifted to near-zero. | `rho_s = head(x) + rho0_s` in model.py |
| 2 | Sigma_rs → ∞ (Kendall pathology) | Total loss went negative. rho_s gradient → 0. | `log_sigma_rs.clamp_(max=0.0)` after each optimizer step |
| 3 | All-sample val MSE for checkpointing | `*` (best) markers after epoch 30 were noise. Early stopping broken. | Switch to positive-only `pos_rho_s_mse` for checkpoint selection |
| 4 | Gate 1 used all-sample baseline | All-sample baseline ~5e-8 (dominated by negatives where rho≈rho0≈0); not a useful signal | Gate 1 and baseline both switched to positive-only |
| 5 | LR=1e-3 too high | Oscillation at epoch 15-30 after reaching good basin | LR 1e-3 → 2e-4; min_lr 1e-5 → 1e-6 |
| 6 | ch3oh as val molecule | 9x worse than baseline — C-O-H alcohol combination absent from training set | Switch to ch3f (isoelectronic with training molecules) |

### The 87% Negative Problem

The MRA tree structure produces ~87% negative (below-leaf) samples per molecule (each internal node generates 8=2³ children, only 1 of which is a true in-tree leaf on average). This means:
- All-sample MSE is dominated by negatives where rho≈rho0≈0
- The model can trivially minimize all-sample MSE by outputting rho0_s, collapsing the metric to near-zero by epoch 30
- Two mitigations applied: (1) `pos_rho_weight=10.0` upweights in-tree samples in the loss; (2) gate and checkpoint selection use positive-only MSE

### Data Quantity Bottleneck

With 13 training molecules and ch3f as val, best positive-only val MSE = 1.375e-6 vs. baseline 4.846e-7 (2.8x worse after 120 epochs, still improving). The model needs more diverse training molecules. The W4-11 dataset (at `/gpfs/projects/rjh/ruhin/perf_pipeline/molecules/W4-11/`) provides standardized geometries for additional molecules. Priority targets: n2o, h2s, ocs, hnco (fill N-O, S, and C=S bond types absent from current training set).

### Convergence

Model ran all 120 epochs without triggering early stopping (still improving at epoch 119). Two paths forward: (1) more training molecules for better generalization, (2) more epochs (120 → 240) for the current molecule set to converge.
