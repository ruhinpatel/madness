# MRA-NN Negative Results Paper — Design Spec

**Date:** 2026-08-22
**Target venue:** JCTC (Journal of Chemical Theory and Computation)
**Narrative arc:** "We tried X, here's why it fails, here's what to do instead" (Arc A). Upgradeable to Arc B (Path 2 results as centerpiece) if Dalton density ML input approach succeeds.

---

## Working Title

"Why Local ML Models Cannot Accelerate SCF Convergence in Multiresolution Analysis: Lessons from Systematic Architecture Search"

## Scope

The paper covers Decisions 1-27 of the MRA-NN project: three ML approaches to SCF acceleration using a FiLM-conditioned MLP with same-level halo neighbors in the MADNESS MRA framework. All three hit the same architectural limit. The paper explains *why* — coarse-level convergence dominance + input signal bottleneck — and extracts transferable lessons for ML-for-adaptive-mesh methods.

## Decision Presentation Strategy

**Curated highlights (Option B):** Group the 27 decisions into key experiments:
1. Multi-task vs single-task ablation (Decisions 7, 12, 15)
2. Level clamping diagnostic (Decision 17)
3. Parent feature concatenation (Decisions 18-22)
4. Refine-only training (Decisions 25-27)

Bug fixes and band-aids (log_sigma clamping, positive-only checkpointing, LR adjustment, validation molecule swap) become footnotes or supplementary material. The full 27-decision audit goes in supplementary.

## Paper Structure

### 1. Introduction (~1 page)
- MADNESS and MRA for DFT — the SCF convergence bottleneck
- ML for initial guess: the promise (fewer iterations via better rho0)
- Our contribution: systematic negative result + mechanistic explanation + path forward

### 2. Background (~1 page)
- MRA tree structure: scaling coefficients, wavelet coefficients, adaptive refinement, key (n,l) addressing
- SCF iteration mechanics: how initial density quality affects convergence path, multi-protocol thresholds
- Prior work: Gong et al. GED-CRN (19-molecule electron density prediction on grids), other ML density prediction approaches in Gaussian basis

### 3. Method (~2 pages)

#### 3.1 Data Pipeline
- MADNESS `dump_training_functions` exports rho0, vnuc, rho_converged to HDF5
- pymra reads HDF5, computes halos (6 face-adjacent same-level neighbors via two-scale relation), generates negatives (8 children below each leaf)
- Dataset: 51 W4-11 molecules, k=8/thresh=1e-6, ~208k samples
- Train/val/test split: 45/3/3 molecules

#### 3.2 Architecture
- MRANet: FiLM-conditioned MLP with factored halo encoder
- Inputs: center rho0_s + vnuc_s (1024), 6 halo neighbor coefficients (6x512 via shared encoder → 768), level embedding (32-dim FiLM conditioning)
- Trunk: 3 FiLM layers (variable width → 256)
- Output: rho_s via residual connection (head(x) + rho0_s)
- What the model sees: 6 same-level face-adjacent neighbors
- What the model does NOT see: parent node, cross-level information, molecular geometry

#### 3.3 Training
- Single-task loss (after multi-task interference diagnosis)
- Level-aware masking: hard cutoff <200 samples, soft sqrt(count/max_count) weighting
- AdamW, lr=2e-4, cosine schedule, AMP, early stopping (patience=20)
- k=8/thresh=1e-6 (MADNESS production precision)

### 4. Experiments & Results (~3 pages)

#### 4.1 Density Prediction
- Per-level MSE breakdown (Figure 2): levels 10-14 beat baseline (0.41-0.98x), coarse levels at parity or worse
- Single-task ablation: removing multi-task heads improved train ratio from 2.99x to 1.00x
- Level 0 (3 samples) accounts for 99.9% of aggregate MSE — masking fixes training but not inference

#### 4.2 Parent Features
- Added parent rho0/vnuc s-coefficients (1024 floats) via trunk concatenation (+1M params)
- Result: model learned to ignore them entirely — per-level breakdown identical to without parents
- Explanation: parent rho0/vnuc encode the same information as child level, just at coarser resolution — no new signal added

#### 4.3 SCF Convergence Test
- 7 molecules x 2 thresholds (1e-6, 1e-8) = 14 comparisons
- Level-clamped model (levels 10-14 only) vs baseline rho0
- Result: 14/14 identical iteration counts (Table/Figure 3)
- Correct electronic state recovered (energy matches to 8 sig figs)
- Conclusion: fine-level accuracy does not drive SCF convergence

#### 4.4 Tree Structure Prediction
- Pivot from density regression to refinement classification (binary)
- Refine-only training (pure focal loss): F1=0.860 ceiling
- Multi-task version: F1=0.857 — removing interference moved F1 by +0.003
- SCF tree-walk test: 31 iterations vs baseline 12 (worse)
- Confirms architectural limit applies to both regression and classification

### 5. Analysis: Why Coarse Levels Dominate (~1.5 pages)
- SCF convergence determined by coarse-level (1-9) density quality
- Fine-level improvements (levels 10-14) are projected away in first protocol step (thresh=1e-4)
- Tighter thresholds (1e-8) don't change this — additional refinement at finer levels doesn't alter coarse-level convergence path
- The information bottleneck: rho0/vnuc at coarse levels are smooth sums of atomic densities — the density correction (exchange-correlation, orbital relaxation, bonding) requires non-local electron-electron interaction information not present in these inputs
- Parent features fail because they encode the same signal at coarser resolution, not the *correction* signal (which is what we're trying to predict)
- Architecture vs. input: no architecture (MLP, GNN, transformer) can extract information that isn't in the input

### 6. Lessons and Path Forward (~1 page)

#### Transferable Lessons
1. **Input signal quality > architecture**: If the input doesn't contain the target signal, no model capacity or receptive field expansion will help. Verify input sufficiency before scaling architecture.
2. **Fine-level accuracy != solver acceleration**: In multi-scale iterative solvers, identify which scale drives convergence before optimizing predictions at other scales.
3. **Multi-task interference is insidious**: Uncertainty-weighted multi-task losses can silently suppress the hardest task. Verify loss function matches stated objective by tracing the code path, not trusting config flag names.

#### Path Forward: Dalton Density as ML Input
- Gaussian basis set calculations (Dalton) provide a fundamentally richer starting point than promolecular density
- Preliminary results from collaborator: direct Dalton density already reduces MADNESS SCF iterations and can skip coarse protocols
- ML opportunity: predict the small residual (Dalton density → MRA-converged density) rather than the large correction (rho0 → rho)
- The residual is orders of magnitude smaller, potentially tractable for the existing local architecture

### 7. Conclusion (~0.5 page)
- Summary of findings
- The negative result as a positive contribution: saves other groups from repeating this path
- Open question: can Dalton+ML bridge the input signal gap?

### Supplementary Material
- Full 27-decision audit table (all decisions with dates, commits, results, verdicts)
- Training hyperparameters and ablation details (all 6+ training runs)
- Per-molecule SCF iteration count table (all 14 comparisons)
- Density comparison details (L2 norms, per-leaf MSE, dipole moments)

## Figures

1. **Architecture diagram**: MRANet with halo neighbors, FiLM conditioning, residual connection. Annotate what the model sees vs. what it needs.
2. **Per-level MSE breakdown**: Bar chart showing ratio (model MSE / baseline MSE) per tree level. Key visual: levels 10-14 below 1.0, coarse levels above.
3. **SCF convergence table/figure**: 7 molecules x 2 thresholds, all showing identical iteration counts. The "14/14" result.
4. **MRA tree schematic**: Illustrate same-level neighbors (what model sees) vs. multi-scale context (what SCF convergence requires). Show information flow needed at coarse levels.

## Estimated Length
~10 pages in JCTC format (standard for methods/negative-results papers)

## Key Source Files (on SeaWulf)
| File | Role in paper |
|------|---------------|
| `mra_nn/docs/2026-08-10-scf-test-postmortem.md` | Raw material for Sections 4-5, Supplementary decision table |
| `mra_nn/model.py` | Architecture description (Section 3.2) |
| `mra_nn/losses.py` | Loss functions (Section 3.3) |
| `mra_nn/train.py` | Training loop details |
| `mra_nn/compare_densities.py` | Density comparison methodology (Section 4.3) |
| `mra_nn/slurm/scf_test.sh` | SCF test methodology (Section 4.3) |
| `mra_nn/slurm/scf_test_multi.sh` | Multi-molecule SCF test (Section 4.3) |
| `mra_nn/dataset_builder.py` | Data pipeline description (Section 3.1) |
| `mra_nn/configs/single_task.yaml` | Training config reference |
| `mra_nn/configs/refine_only.yaml` | Refine-only config reference |

## Upgrade Path (Arc A → Arc B)
If Path 2 (Dalton density as ML input) produces positive results:
- Section 6 expands from "future work" to a full Results section
- Paper reframes: negative results become motivation, Path 2 results become the contribution
- Title changes to emphasize the solution, not the failure
- Supplementary absorbs more of the decision audit detail
