# Step A: Inference-Time Level Clamping + SCF Test

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Model:** Sonnet 4.6 for all tasks.

**Goal:** Add level clamping to inference so the model outputs rho0 at levels where it produces garbage, then re-run the SCF convergence test to see if the 2.9% density improvement at levels 10-14 translates to fewer SCF iterations.

**Architecture:** Modify `predict_density_simple()` in `predict.py` to accept a `use_model_levels` set. At any level outside this set, the predicted s-coefficients are replaced with rho0_s (identity through the residual connection). The SCF test script is updated to use the Option A checkpoint and corrected PYTHONPATH. No retraining.

**Tech Stack:** Python (PyTorch, pymra), MADNESS C++ (existing binaries), SLURM

## Global Constraints

- `PYTHONPATH` must include both `/gpfs/projects/rjh/ruhin/madness-ruhin` (for `mra_nn` package) and `/gpfs/projects/rjh/adrian/pymra/src` (for `pymra`)
- `MAD_NUM_THREADS` must be set to `ntasks - 1`
- Checkpoint path: `/gpfs/projects/rjh/ruhin/mra_nn/checkpoints/2026-08-10_06-02/best.pt`
- Test molecule: ch3oh (methanol), 18 electrons
- Baseline SCF result from previous test: energy -114.850 Ha, 10 total iterations, dipole 0.645 a.u.
- Never add `Co-Authored-By: Claude` lines to commit messages

---

### Task 1: Add level clamping to `predict.py` and update `scf_test.sh`

**Files:**
- Modify: `mra_nn/predict.py:148-217` (`predict_density_simple` function)
- Modify: `mra_nn/predict.py:302-349` (`main` function — add CLI arg)
- Modify: `mra_nn/slurm/scf_test.sh:34,136-140` (PYTHONPATH + checkpoint path)

**Interfaces:**
- Consumes: existing `predict_density_simple()` signature
- Produces: `predict_density_simple()` gains `use_model_levels: set[int] | None` parameter (default None = use model at all levels, for backward compat). CLI gains `--use-model-levels` argument.

- [ ] **Step 1: Add `use_model_levels` parameter to `predict_density_simple()`**

In `mra_nn/predict.py`, modify the function signature and the inner loop where predictions are written to the tree. When a leaf's level is NOT in `use_model_levels`, copy rho0's s-coefficients instead of the model prediction.

```python
@torch.no_grad()
def predict_density_simple(
    model: MRANet,
    rho0_path: str,
    vnuc_path: str,
    n_electrons: int,
    device: torch.device,
    batch_size: int = 4096,
    use_model_levels: set[int] | None = None,
) -> FunctionTree:
```

In the inner loop (line ~208), replace:

```python
        for i, key in enumerate(batch_keys):
            predicted_tree.nodes[key] = Node(
                s=rho_s_np[i].reshape((k,) * ndim).astype(np.float64)
            )
```

with:

```python
        for i, key in enumerate(batch_keys):
            if use_model_levels is not None and key.n not in use_model_levels:
                # Outside effective range — use rho0 as-is
                predicted_tree.nodes[key] = Node(
                    s=node_s(rho0_tree, key).copy()
                )
            else:
                predicted_tree.nodes[key] = Node(
                    s=rho_s_np[i].reshape((k,) * ndim).astype(np.float64)
                )
```

- [ ] **Step 2: Add `--use-model-levels` CLI argument to `main()`**

In the `main()` function, add the argument and pass it through:

```python
    parser.add_argument(
        "--use-model-levels", type=str, default=None,
        help="Comma-separated levels where model predictions are used (e.g. '10,11,12,13,14'). "
             "At other levels, rho0 is used unchanged. Default: use model at all levels.",
    )
```

Parse it before calling `predict_density_simple`:

```python
    use_model_levels = None
    if args.use_model_levels:
        use_model_levels = set(int(x) for x in args.use_model_levels.split(","))
        print(f"Level clamping: using model at levels {sorted(use_model_levels)}, rho0 elsewhere")
```

Pass to the single-task call:

```python
    if single_task:
        tree = predict_density_simple(
            model, args.rho0, args.vnuc,
            n_electrons=args.n_electrons,
            device=device,
            use_model_levels=use_model_levels,
        )
```

- [ ] **Step 3: Update `slurm/scf_test.sh` — fix PYTHONPATH, checkpoint, and add level clamping**

In `mra_nn/slurm/scf_test.sh`:

Line 34 — fix PYTHONPATH to include madness-ruhin:
```bash
export PYTHONPATH=/gpfs/projects/rjh/ruhin/madness-ruhin:/gpfs/projects/rjh/adrian/pymra/src:${PYTHONPATH:-}
```

Lines 136-140 — update checkpoint to Option A model and add level clamping:
```bash
$PYTHON mra_nn/predict.py \
    --checkpoint "$DATA/checkpoints/2026-08-10_06-02/best.pt" \
    --rho0 "$DATA/training_data/ch3oh/rho0.mad.h5" \
    --vnuc "$DATA/training_data/ch3oh/vnuc.mad.h5" \
    --n-electrons 18 \
    --use-model-levels "10,11,12,13,14" \
    --out "$TESTDIR/rhoML.mad.h5"
```

- [ ] **Step 4: Dry-run the test script**

```bash
/cm/shared/apps/slurm/21.08.8/bin/sbatch --test-only /gpfs/projects/rjh/ruhin/madness-ruhin/mra_nn/slurm/scf_test.sh
```

Expected: `Job NNNNN to start at ...` (no errors).

- [ ] **Step 5: Commit**

```bash
cd /gpfs/projects/rjh/ruhin/madness-ruhin
git add mra_nn/predict.py mra_nn/slurm/scf_test.sh
git commit -m "feat(mra-nn): add inference-time level clamping to predict.py

Level clamping forces model to output rho0 at levels outside a
specified set. Diagnostic showed levels 10-14 beat baseline (0.41-
0.98x) while level 0 produces 1.3Bx worse predictions due to
training-time masking. Clamped model gives 0.971x on held-out val.

Also fixes scf_test.sh: PYTHONPATH now includes madness-ruhin for
mra_nn package, checkpoint updated to Option A model."
```

---

### Task 2: Submit SCF test and analyze results

**Files:**
- Execute: `mra_nn/slurm/scf_test.sh` (submit via sbatch)
- Read: `mra_nn/logs/scf_test_<jobid>.out` (results)
- No code changes

**Interfaces:**
- Consumes: level-clamped `predict.py` from Task 1
- Produces: SCF iteration count, energy, dipole for baseline vs. level-clamped ML guess

- [ ] **Step 1: Submit the SCF test**

```bash
/cm/shared/apps/slurm/21.08.8/bin/sbatch /gpfs/projects/rjh/ruhin/madness-ruhin/mra_nn/slurm/scf_test.sh
```

Monitor with:
```bash
/cm/shared/apps/slurm/21.08.8/bin/squeue -u ruhipatel
```

Job runs ~30-60 min (build + 2 SCF runs on 1 core).

- [ ] **Step 2: Read output and compare against known baseline**

Once job completes, read the log:
```bash
cat /gpfs/projects/rjh/ruhin/mra_nn/logs/scf_test_<jobid>.out
```

Extract and compare:

| Metric | Baseline (rho0) | ML clamped [10-14] | Previous ML (no clamp) |
|--------|-----------------|---------------------|------------------------|
| Final energy (Ha) | -114.850 | ? | -114.263 (WRONG STATE) |
| Total iterations | 10 | ? | 30 |
| Dipole (a.u.) | 0.645 | ? | 2.015 (WRONG) |

**Success:** Energy within 1e-3 Ha of -114.850 AND iterations < 10.
**Neutral:** Same state, same iterations (fine-level accuracy doesn't drive SCF).
**Failure:** Wrong state again (investigate — shouldn't happen since coarse levels are rho0).

- [ ] **Step 3: Run density comparison**

```bash
cd /gpfs/projects/rjh/ruhin/madness-ruhin
source /gpfs/projects/rjh/ruhin/mra_nn/.venv/bin/activate
export PYTHONPATH=/gpfs/projects/rjh/ruhin/madness-ruhin:/gpfs/projects/rjh/adrian/pymra/src:${PYTHONPATH:-}

python mra_nn/compare_densities.py \
    --rho0 /gpfs/projects/rjh/ruhin/mra_nn/training_data/ch3oh/rho0.mad.h5 \
    --rho-conv /gpfs/projects/rjh/ruhin/mra_nn/training_data/ch3oh/rho.mad.h5 \
    --rho-ml /gpfs/projects/rjh/ruhin/mra_nn/scf_test/ch3oh/rhoML.mad.h5
```

Expected: L2 ratio (ML/rho0) should be < 1.0 (consistent with the 0.971x diagnostic).

- [ ] **Step 4: Update CLAUDE.md and Notion with results**

Add a dated entry to `mra_nn/CLAUDE.md` under "Job Chain" with the SCF test results. Update the Notion page's Progress & Meeting Log.

- [ ] **Step 5: Decision gate**

Based on results, determine next step per the spec:
- Iterations decreased → Step A succeeded. Proceed to Step B for more improvement.
- Iterations unchanged → Fine-level accuracy alone doesn't drive SCF. Step B must unlock coarse levels.
- Wrong state → Investigate before Step B.

Document the decision in CLAUDE.md.
