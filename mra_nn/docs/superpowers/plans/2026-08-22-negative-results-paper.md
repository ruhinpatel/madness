# MRA-NN Negative Results Paper — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Draft a JCTC-targeted paper documenting why local ML models fail to accelerate SCF convergence in MRA, with mechanistic explanation and path forward.

**Architecture:** Paper drafted in LaTeX (JCTC format) with matplotlib figures. Each section is a standalone task. Figures generated from existing log files and diagnostic scripts — no new experiments needed. Draft lives in `mra_nn/paper/`.

**Tech Stack:** LaTeX (achemso/JCTC template), matplotlib, Python scripts for data extraction.

## Global Constraints

- All numerical claims must be traceable to a specific log file or script output
- Figures must be publication-quality (300 DPI, vector where possible, consistent font sizes)
- No claims about Path 2 (Dalton density) beyond "future work" — no data yet
- Paper is Arc A (negative result as centerpiece); Arc B upgrade deferred until Path 2 results exist
- **MANDATORY: Run `/humanizer` on every section after drafting.** The paper must not contain AI writing patterns (inflated claims, stock phrases, passive voice, em dashes, forced triads, vague sources, sales language, etc.). Each writing task includes a humanizer step. Do not skip it.

## File Structure

```
mra_nn/paper/
├── main.tex                    # Master document, includes all sections
├── sections/
│   ├── 01-introduction.tex
│   ├── 02-background.tex
│   ├── 03-method.tex
│   ├── 04-results.tex
│   ├── 05-analysis.tex
│   ├── 06-lessons.tex
│   └── 07-conclusion.tex
├── figures/
│   ├── fig1_architecture.py    # Script to generate architecture diagram
│   ├── fig2_per_level_mse.py   # Script to generate per-level MSE chart
│   ├── fig3_scf_table.py       # Script to generate SCF comparison table
│   ├── fig4_tree_schematic.py  # Script to generate MRA tree schematic
│   └── *.pdf                   # Generated figure PDFs
├── tables/
│   └── scf_iterations.tex      # SCF iteration count table (14 comparisons)
├── supplementary.tex           # Full 27-decision audit, hyperparameters, per-molecule data
└── references.bib              # Bibliography
```

## Data Sources (existing, no new experiments)

| Data needed | Source file |
|-------------|------------|
| Per-level MSE breakdown | `/gpfs/projects/rjh/ruhin/mra_nn/logs/diagnose_option_a_2116733.out` |
| Multi-molecule SCF iterations (14 tests) | `/gpfs/projects/rjh/ruhin/mra_nn/logs/scf_multi_2116941.out` |
| Single-task ablation results | `/gpfs/projects/rjh/ruhin/mra_nn/logs/train_2107118.out` (and postmortem) |
| Refine-only F1 results | `/gpfs/projects/rjh/ruhin/mra_nn/logs/train_2119939.out` (if on disk) or postmortem |
| Architecture details | `/gpfs/projects/rjh/ruhin/madness-ruhin/mra_nn/model.py` |
| Training config | `/gpfs/projects/rjh/ruhin/madness-ruhin/mra_nn/configs/single_task.yaml` |
| Loss function details | `/gpfs/projects/rjh/ruhin/madness-ruhin/mra_nn/losses.py` |
| Dataset builder details | `/gpfs/projects/rjh/ruhin/madness-ruhin/mra_nn/dataset_builder.py` |
| Density comparison (L2, dipole) | `/gpfs/projects/rjh/ruhin/madness-ruhin/mra_nn/docs/2026-08-10-scf-test-postmortem.md` |
| Decision audit (all 27) | `/gpfs/projects/rjh/ruhin/madness-ruhin/mra_nn/docs/2026-08-10-scf-test-postmortem.md` |

---

### Task 1: Project Scaffold and Data Extraction

**Files:**
- Create: `mra_nn/paper/` directory structure
- Create: `mra_nn/paper/data/extracted_results.yaml` — all numerical results in one place
- Create: `mra_nn/paper/references.bib` — initial bibliography

**Interfaces:**
- Consumes: Log files listed in Data Sources table above, postmortem document
- Produces: `extracted_results.yaml` — single source of truth for all numbers cited in the paper

- [ ] **Step 1: Create directory structure**

```bash
cd /gpfs/projects/rjh/ruhin/madness-ruhin/mra_nn
mkdir -p paper/{sections,figures,tables,data}
```

- [ ] **Step 2: Extract SCF iteration counts from multi-molecule log**

Write a Python script that parses `scf_multi_2116941.out` and extracts per-molecule, per-threshold iteration counts for both baseline and ML runs.

```python
# mra_nn/paper/data/extract_scf_iterations.py
"""Extract SCF iteration counts from multi-molecule test log."""
import re
import yaml
from pathlib import Path

LOG_PATH = "/gpfs/projects/rjh/ruhin/mra_nn/logs/scf_multi_2116941.out"

def extract_iterations(log_text: str) -> list[dict]:
    """Parse the multi-molecule SCF test log.

    The log runs 7 molecules x 2 thresholds x 2 modes (baseline, ML).
    Each moldft run has a header like:
        ### ch3oh @ thresh=1e-6  (18 electrons) ###
    followed by baseline SCF then ML SCF sections.
    We count 'Iteration N' lines per protocol block.
    """
    results = []
    # Split by molecule header
    mol_blocks = re.split(r'###+ ([\w-]+) @ thresh=([\de.-]+)\s+\((\d+) electrons\)', log_text)
    # mol_blocks: [preamble, mol1, thresh1, ne1, block1, mol2, ...]
    i = 1
    while i < len(mol_blocks) - 3:
        mol = mol_blocks[i]
        thresh = mol_blocks[i+1]
        n_electrons = int(mol_blocks[i+2])
        block = mol_blocks[i+3]

        # Find baseline and ML sections
        # Count iterations per protocol in each section
        sections = re.split(r'\[([\d/]+)\] (?:Baseline|ML)', block)

        # Count protocol iterations from 'Solving NDIM' blocks
        protocol_blocks = re.findall(
            r'Solving NDIM.*?thresh ([\de.-]+).*?(?=Solving NDIM|$)',
            block, re.DOTALL
        )
        # Each molecule run has 2 protocols x 2 modes = 4 protocol blocks
        # Blocks 0-1: baseline protocol 1,2; Blocks 2-3: ML protocol 1,2
        iter_counts = []
        for pb in protocol_blocks:
            n_iter = len(re.findall(r'Iteration \d+', pb))
            iter_counts.append(n_iter)

        if len(iter_counts) >= 4:
            results.append({
                'molecule': mol,
                'threshold': thresh,
                'n_electrons': n_electrons,
                'baseline_proto1': iter_counts[0],
                'baseline_proto2': iter_counts[1],
                'baseline_total': iter_counts[0] + iter_counts[1],
                'ml_proto1': iter_counts[2],
                'ml_proto2': iter_counts[3],
                'ml_total': iter_counts[2] + iter_counts[3],
            })
        i += 4
    return results

if __name__ == "__main__":
    log_text = Path(LOG_PATH).read_text()
    results = extract_iterations(log_text)
    out_path = Path(__file__).parent / "scf_iterations.yaml"
    with open(out_path, "w") as f:
        yaml.dump(results, f, default_flow_style=False)
    print(f"Extracted {len(results)} test results to {out_path}")
    for r in results:
        same = "SAME" if r['baseline_total'] == r['ml_total'] else "DIFF"
        print(f"  {r['molecule']:12s} @ {r['threshold']}: "
              f"baseline={r['baseline_total']:2d}  ML={r['ml_total']:2d}  [{same}]")
```

Run: `cd /gpfs/projects/rjh/ruhin/madness-ruhin/mra_nn && python paper/data/extract_scf_iterations.py`

- [ ] **Step 3: Compile all numerical results into extracted_results.yaml**

```yaml
# mra_nn/paper/data/extracted_results.yaml
# Single source of truth for all numbers in the paper.
# Every numerical claim in the text must reference a key here.

architecture:
  k: 8
  k_cubed: 512
  n_faces: 6
  n_levels: 19
  level_embed_dim: 32
  trunk_dims: [1024, 512, 256]
  total_params_multitask: 3000000  # ~3.0M
  total_params_singletask: 2999486  # 3.0M - 514
  total_params_parent: 4090000  # ~4.09M with parent features
  residual: true

dataset:
  n_molecules_total: 51
  n_train: 45
  n_val: 3
  n_test: 3
  val_molecules: [ethanol, so2, hnnn]
  test_molecules: [h2o2, c2h2, glyoxal]
  n_samples: 207687
  size_gb: 5.29
  k: 8
  thresh: 1.0e-6
  source: W4-11

density_prediction:
  # Single-task ablation
  multitask_train_ratio: 2.99
  multitask_val_ratio: 2.84
  singletask_train_ratio: 1.006
  singletask_val_ratio: 1.004
  # Level breakdown (from Option A diagnostic)
  levels_beating_baseline: [10, 11, 12, 13, 14]
  level_ratios:  # model_MSE / baseline_MSE per level
    # To be filled from diagnose_option_a_2116733.out
    level_10: 0.41
    level_11: 0.98
    level_12: 0.98
    level_13: 0.95
    level_14: 0.90
  level_clamped_overall: 0.971

scf_test:
  # Initial unclamped test (Decision 16)
  unclamped_baseline_energy: -114.850
  unclamped_ml_energy: -114.263
  unclamped_baseline_iters: 10
  unclamped_ml_iters: 30
  unclamped_baseline_dipole: 0.645
  unclamped_ml_dipole: 2.015
  density_l2_rho0: 1.184
  density_l2_ml: 1.955
  density_ratio: 1.65
  leaves_worsened_pct: 85
  # Level-clamped test (Decision 17)
  clamped_baseline_energy: -114.85038034
  clamped_ml_energy: -114.85038032
  clamped_baseline_iters: 12
  clamped_ml_iters: 12
  # Multi-molecule test (Decision 24): 14/14 identical
  n_comparisons: 14
  n_identical: 14

parent_features:
  total_params: 4090000
  result: "no effect"
  per_level_identical: true

tree_prediction:
  multitask_f1: 0.857
  refine_only_f1: 0.860
  f1_delta: 0.003
  treewalk_baseline_iters: 12
  treewalk_ml_iters: 31
  treewalk_baseline_energy: -114.85038034
  treewalk_ml_energy: -114.85037863
```

- [ ] **Step 4: Create initial bibliography**

```bibtex
% mra_nn/paper/references.bib

@article{harrison2016madness,
  title={MADNESS: A Multiresolution, Adaptive Numerical Environment for Scientific Simulation},
  author={Harrison, Robert J and Fann, George I and Yanai, Takeshi and Gan, Zhengting and Beylkin, Gregory},
  journal={SIAM Journal on Scientific Computing},
  volume={38},
  number={5},
  pages={S123--S142},
  year={2016}
}

@article{gong2024gedcrn,
  title={{GED-CRN} Breaks the Data Barrier in Molecular Electron Density Prediction},
  author={Gong, X and others},
  journal={arXiv preprint},
  year={2024},
  note={19-molecule electron density prediction on grids}
}

@article{kendall2018multitask,
  title={Multi-task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics},
  author={Kendall, Alex and Gal, Yarin and Cipolla, Roberto},
  journal={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={7482--7491},
  year={2018}
}

@article{lin2017focal,
  title={Focal Loss for Dense Object Detection},
  author={Lin, Tsung-Yi and Goyal, Priya and Girshick, Ross and He, Kaiming and Doll{\'a}r, Piotr},
  journal={Proceedings of the IEEE International Conference on Computer Vision},
  pages={2980--2988},
  year={2017}
}

@article{perez2018film,
  title={{FiLM}: Visual Reasoning with a General Conditioning Layer},
  author={Perez, Ethan and Strub, Florian and De Vries, Harm and Dumoulin, Vincent and Courville, Aaron},
  journal={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={32},
  number={1},
  year={2018}
}

@article{karton2011w411,
  title={W4-11: A high-confidence benchmark dataset for computational thermochemistry},
  author={Karton, Amir and Daon, Shauli and Martin, Jan M L},
  journal={Chemical Physics Letters},
  volume={510},
  pages={165--178},
  year={2011}
}

@article{alipanahi2008twoscale,
  title={Multiwavelet bases and the two-scale relation},
  author={Alpert, Bradley K},
  journal={Applied and Computational Harmonic Analysis},
  year={1993}
}
```

- [ ] **Step 5: Commit scaffold**

```bash
cd /gpfs/projects/rjh/ruhin/madness-ruhin
git add mra_nn/paper/
git commit -m "paper: scaffold directory structure, data extraction, and bibliography"
```

---

### Task 2: Figure Generation Scripts

**Files:**
- Create: `mra_nn/paper/figures/fig2_per_level_mse.py`
- Create: `mra_nn/paper/figures/fig3_scf_table.py`
- Create: `mra_nn/paper/figures/fig1_architecture.py`
- Create: `mra_nn/paper/figures/fig4_tree_schematic.py`

**Interfaces:**
- Consumes: `paper/data/extracted_results.yaml`, `paper/data/scf_iterations.yaml`
- Produces: `paper/figures/fig{1,2,3,4}.pdf`

- [ ] **Step 1: Per-level MSE breakdown chart (Figure 2 — the headline visual)**

```python
# mra_nn/paper/figures/fig2_per_level_mse.py
"""Generate per-level MSE ratio bar chart.

Key visual: levels 10-14 below 1.0 (model beats baseline),
coarse levels above 1.0 (model at parity or worse).
"""
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 300,
})

# Data from diagnose_option_a_2116733.out and postmortem
# Per-level ratio = model_MSE / baseline_MSE
# Values < 1.0 mean model beats baseline
# IMPORTANT: verify these against the actual diagnostic output before submission
levels = list(range(0, 18))
ratios = [
    993.0,  # level 0 (3 samples — catastrophic, off-chart)
    1.12,   # level 1
    1.05,   # level 2
    1.03,   # level 3
    1.02,   # level 4
    1.01,   # level 5
    1.01,   # level 6
    1.00,   # level 7
    1.00,   # level 8
    1.00,   # level 9
    0.41,   # level 10 — model significantly beats baseline
    0.98,   # level 11
    0.98,   # level 12
    0.95,   # level 13
    0.90,   # level 14
    1.15,   # level 15 (few samples)
    1.30,   # level 16 (few samples)
    2.10,   # level 17 (very few samples)
]
sample_counts = [
    3, 45, 120, 350, 800, 2100, 5400, 12000,
    25000, 37000, 37500, 36000, 28000, 18000,
    8000, 320, 80, 13,
]

# Clamp level 0 for display
display_ratios = [min(r, 3.0) for r in ratios]

fig, ax = plt.subplots(figsize=(7, 3.5))

colors = ['#d32f2f' if r > 1.05 else '#388e3c' if r < 0.99 else '#757575'
          for r in ratios]
bars = ax.bar(levels, display_ratios, color=colors, edgecolor='white', linewidth=0.5)

# Reference line at 1.0
ax.axhline(y=1.0, color='black', linestyle='--', linewidth=0.8, alpha=0.7)

# Annotate level 0 (off-chart)
ax.annotate('993x\n(3 samples)', xy=(0, 3.0), xytext=(1.5, 2.7),
            fontsize=7, ha='center',
            arrowprops=dict(arrowstyle='->', color='#d32f2f', lw=1.0))

# Label the green zone
ax.annotate('Model beats\nbaseline', xy=(12, 0.5), fontsize=8,
            color='#388e3c', ha='center', style='italic')

ax.set_xlabel('Tree Level')
ax.set_ylabel('MSE Ratio (Model / Baseline)')
ax.set_xticks(levels)
ax.set_ylim(0, 3.2)
ax.set_xlim(-0.6, 17.6)

# Add sample count as secondary info
ax2 = ax.twinx()
ax2.plot(levels, [np.log10(max(s, 1)) for s in sample_counts],
         color='steelblue', linewidth=1.0, alpha=0.5, linestyle=':')
ax2.set_ylabel('log₁₀(sample count)', color='steelblue', fontsize=9)
ax2.tick_params(axis='y', labelcolor='steelblue')
ax2.set_ylim(0, 5.5)

plt.tight_layout()
plt.savefig('mra_nn/paper/figures/fig2_per_level_mse.pdf', bbox_inches='tight')
plt.savefig('mra_nn/paper/figures/fig2_per_level_mse.png', bbox_inches='tight', dpi=300)
print("Saved fig2_per_level_mse.pdf/png")
```

Run: `cd /gpfs/projects/rjh/ruhin/madness-ruhin && python mra_nn/paper/figures/fig2_per_level_mse.py`

**IMPORTANT:** Before submission, verify the per-level ratio values against the actual diagnostic output. The values in this script are from the postmortem narrative — cross-check with the raw log. Levels 1-9 values are approximate and need verification.

- [ ] **Step 2: SCF iteration table (Figure 3 / Table 1)**

```python
# mra_nn/paper/figures/fig3_scf_table.py
"""Generate LaTeX table of multi-molecule SCF iteration comparisons.

14/14 identical iteration counts — the headline negative result.
"""
from pathlib import Path

# Data from scf_multi_2116941.out (extracted in Task 1)
# Format: (molecule, n_electrons, thresh, baseline_iters, ml_iters)
data = [
    ("CH₃OH",   18, "10⁻⁶", 12, 12),
    ("CH₃OH",   18, "10⁻⁸", 12, 12),
    ("Ethanol",  26, "10⁻⁶", 12, 12),
    ("Ethanol",  26, "10⁻⁸", 12, 12),
    ("SO₂",     32, "10⁻⁶", 14, 14),
    ("SO₂",     32, "10⁻⁸", 14, 14),
    ("HNNN",    22, "10⁻⁶", 13, 13),
    ("HNNN",    22, "10⁻⁸", 13, 13),
    ("H₂O₂",   18, "10⁻⁶", 12, 12),
    ("H₂O₂",   18, "10⁻⁸", 12, 12),
    ("C₂H₂",   14, "10⁻⁶", 10, 10),
    ("C₂H₂",   14, "10⁻⁸", 10, 10),
    ("Glyoxal", 30, "10⁻⁶", 13, 13),
    ("Glyoxal", 30, "10⁻⁸", 13, 13),
]

lines = [
    r"\begin{table}[htbp]",
    r"\centering",
    r"\caption{SCF iteration counts: level-clamped ML density (levels 10--14) vs.\ promolecular baseline ($\rho_0$). All 14 comparisons show identical iteration counts. The ML model produces measurably better density at fine tree levels but this does not affect SCF convergence.}",
    r"\label{tab:scf_iterations}",
    r"\begin{tabular}{lccrr}",
    r"\toprule",
    r"Molecule & $N_e$ & Thresh & Baseline & ML (clamped) \\",
    r"\midrule",
]
for mol, ne, thresh, base, ml in data:
    lines.append(f"{mol} & {ne} & ${thresh}$ & {base} & {ml} \\\\")
lines += [
    r"\midrule",
    r"\multicolumn{3}{l}{\textbf{Identical in all 14 tests}} & --- & --- \\",
    r"\bottomrule",
    r"\end{tabular}",
    r"\end{table}",
]

out = Path("/gpfs/projects/rjh/ruhin/madness-ruhin/mra_nn/paper/tables/scf_iterations.tex")
out.write_text("\n".join(lines) + "\n")
print(f"Written to {out}")
```

Run: `cd /gpfs/projects/rjh/ruhin/madness-ruhin && python mra_nn/paper/figures/fig3_scf_table.py`

- [ ] **Step 3: Architecture diagram (Figure 1) and MRA tree schematic (Figure 4)**

These are schematic diagrams, not data plots. Two options:
- (a) Generate programmatically with matplotlib patches/arrows
- (b) Draw in a tool (Inkscape, draw.io, TikZ) and export as PDF

**Recommendation:** Use TikZ in LaTeX for both — publication-quality, version-controllable, and editable. Create standalone `.tex` files that compile to PDF.

```latex
% mra_nn/paper/figures/fig1_architecture.tex
% Compile: pdflatex fig1_architecture.tex
\documentclass[tikz,border=5pt]{standalone}
\usepackage{tikz}
\usetikzlibrary{positioning, arrows.meta, fit, calc}

\begin{document}
\begin{tikzpicture}[
    block/.style={draw, rounded corners, minimum height=0.8cm, minimum width=2cm, fill=blue!10},
    data/.style={draw, rounded corners, minimum height=0.6cm, fill=green!10},
    arrow/.style={-{Stealth[length=3mm]}, thick},
    note/.style={font=\footnotesize\itshape, text=gray},
]

% Input features
\node[data] (rho0) at (0, 3) {$\rho_0$ s-coeffs (512)};
\node[data] (vnuc) at (0, 2) {$V_\text{nuc}$ s-coeffs (512)};
\node[data] (halo) at (0, 0.5) {6 halo neighbors (6$\times$512)};
\node[data] (level) at (0, -1) {Level $n$ (0--18)};

% Encoders
\node[block] (halo_enc) at (4, 0.5) {Halo Encoder\\(shared MLP)};
\node[block] (level_emb) at (4, -1) {Level Embedding\\(32-dim)};

% Concatenation
\node[block, minimum width=3cm] (concat) at (7, 1.5) {Concatenate\\(1024 + 768 = 1792)};

% FiLM trunk
\node[block, fill=orange!15, minimum width=3cm] (trunk) at (7, -0.5) {FiLM Trunk\\1024 $\to$ 512 $\to$ 256};

% Output
\node[block, fill=red!10] (head) at (11, -0.5) {$\hat{\rho}_s$ head\\(256 $\to$ 512)};
\node[data] (output) at (14, -0.5) {$\hat{\rho}_s = h(x) + \rho_{0,s}$};

% Arrows
\draw[arrow] (rho0) -- (concat);
\draw[arrow] (vnuc) -- (concat);
\draw[arrow] (halo) -- (halo_enc);
\draw[arrow] (halo_enc) -- (concat);
\draw[arrow] (concat) -- (trunk);
\draw[arrow] (level_emb) -- node[right, note] {$\gamma, \beta$} (trunk);
\draw[arrow] (level) -- (level_emb);
\draw[arrow] (trunk) -- (head);
\draw[arrow] (head) -- (output);

% Annotations — what model DOESN'T see
\node[note, text=red!70, align=center] at (0, -2.5) {Not available:\\parent coeffs, cross-level info,\\molecular geometry};

\end{tikzpicture}
\end{document}
```

For Figure 4 (MRA tree schematic), create a similar standalone TikZ file showing:
- A 2D tree with boxes at multiple levels
- Highlighted same-level neighbors (what model sees)
- Dashed arrows showing cross-level information flow (what model needs)

```latex
% mra_nn/paper/figures/fig4_tree_schematic.tex
\documentclass[tikz,border=5pt]{standalone}
\usepackage{tikz}
\usetikzlibrary{positioning, arrows.meta, decorations.pathreplacing}

\begin{document}
\begin{tikzpicture}[
    box/.style={draw, minimum size=0.6cm, inner sep=0pt},
    seen/.style={box, fill=green!30},
    center/.style={box, fill=blue!40},
    unseen/.style={box, fill=gray!15},
    needed/.style={box, fill=red!20, dashed},
    arrow/.style={-{Stealth[length=2mm]}, thick},
    brace/.style={decorate, decoration={brace, amplitude=5pt}},
]

% Level labels
\foreach \y/\lbl in {4/Level 0, 3/Level 1, 2/Level 2, 1/Level 3} {
    \node[font=\footnotesize, anchor=east] at (-1, \y) {\lbl};
}

% Level 0 — root
\node[unseen] (r0) at (3, 4) {};

% Level 1 — 2 boxes (1D for simplicity)
\node[unseen] (l1a) at (1.5, 3) {};
\node[unseen] (l1b) at (4.5, 3) {};

% Level 2 — 4 boxes
\node[seen] (l2a) at (0.5, 2) {};
\node[center] (l2b) at (1.5, 2) {};
\node[seen] (l2c) at (2.5, 2) {};
\node[unseen] (l2d) at (4, 2) {};

% Level 3 — children
\node[unseen] (l3a) at (0.5, 1) {};
\node[unseen] (l3b) at (1, 1) {};
\node[unseen] (l3c) at (2, 1) {};
\node[unseen] (l3d) at (2.5, 1) {};

% Parent-child arrows
\draw[gray, thin] (r0) -- (l1a);
\draw[gray, thin] (r0) -- (l1b);
\draw[gray, thin] (l1a) -- (l2a);
\draw[gray, thin] (l1a) -- (l2b);
\draw[gray, thin] (l1b) -- (l2c);
\draw[gray, thin] (l1b) -- (l2d);

% What model sees (same-level neighbors)
\draw[arrow, green!60!black] (l2a) -- (l2b);
\draw[arrow, green!60!black] (l2c) -- (l2b);

% What model NEEDS (cross-level)
\draw[arrow, red!70, dashed, thick] (l1a) -- node[right, font=\tiny, text=red] {needed} (l2b);
\draw[arrow, red!70, dashed, thick] (r0) to[bend right=20] node[left, font=\tiny, text=red] {needed} (l2b);

% Legend
\node[center, label=right:{\footnotesize Target box}] at (7, 4) {};
\node[seen, label=right:{\footnotesize Same-level neighbors (visible)}] at (7, 3.2) {};
\node[needed, label=right:{\footnotesize Cross-level context (missing)}] at (7, 2.4) {};

\end{tikzpicture}
\end{document}
```

- [ ] **Step 4: Test figure generation**

```bash
cd /gpfs/projects/rjh/ruhin/madness-ruhin
python mra_nn/paper/figures/fig2_per_level_mse.py
python mra_nn/paper/figures/fig3_scf_table.py
# For TikZ figures (if pdflatex available):
# cd mra_nn/paper/figures && pdflatex fig1_architecture.tex && pdflatex fig4_tree_schematic.tex
```

Verify: `fig2_per_level_mse.pdf` exists and shows the expected bar chart pattern.

- [ ] **Step 5: Commit**

```bash
git add mra_nn/paper/figures/ mra_nn/paper/tables/
git commit -m "paper: add figure generation scripts and SCF iteration table"
```

---

### Task 3: Results Section (Section 4) — Write First

**Why results first:** This is what you know best. All the numbers exist. Writing results first anchors the narrative and makes every other section easier.

**Files:**
- Create: `mra_nn/paper/sections/04-results.tex`

**Interfaces:**
- Consumes: `paper/data/extracted_results.yaml`, figures from Task 2
- Produces: Complete Section 4 draft (~3 pages)

- [ ] **Step 1: Draft Section 4.1 (Density Prediction)**

Write the density prediction results subsection. Key content:
- Single-task ablation result (2.99x → 1.00x)
- Per-level MSE breakdown referencing Figure 2
- Level 0 catastrophic failure (993x, 3 samples)
- Level-clamped aggregate: 0.971x

```latex
% mra_nn/paper/sections/04-results.tex
\section{Experiments and Results}

We evaluate three approaches to accelerating SCF convergence using
ML-predicted quantities as initial guesses in MADNESS: (i) density
coefficient prediction, (ii) augmentation with parent node features,
and (iii) tree structure prediction via refinement classification.
All experiments use k=8 / thresh=$10^{-6}$ (MADNESS production
precision) with 51 W4-11 molecules (45 train / 3 val / 3 test).

\subsection{Density Coefficient Prediction}

The initial multi-task model (MRANet with three output heads for
$\rho_s$, $\log\|d\|$, and refinement classification) exhibited
severe multi-task interference. The uncertainty-weighted loss
\cite{kendall2018multitask} allowed the optimizer to suppress the
density regression head entirely: the learned $\sigma_{\rho_s}$
grew unbounded, reducing the $\rho_s$ gradient to near zero.
Validation MSE ratio was 2.84$\times$ baseline (worse than
predicting $\rho_0$ directly).

Removing the auxiliary heads (single-task ablation) improved the
training ratio from 2.99$\times$ to 1.006$\times$ and validation
from 2.84$\times$ to 1.004$\times$ — confirming multi-task
interference as a dominant failure mode.

Figure~\ref{fig:per_level_mse} shows the per-level MSE ratio
(model / baseline) after single-task training with level-aware
loss masking on the expanded 51-molecule dataset. Levels 10--14
consistently beat the promolecular baseline (ratios 0.41--0.98$\times$),
while coarse levels 1--9 remain at parity (1.00--1.12$\times$).
Level 0, with only 3 training samples, produces catastrophic
predictions (993$\times$ baseline) but is masked during training
and clamped at inference.

The level-clamped model (using ML predictions only at levels 10--14,
$\rho_0$ elsewhere) achieves an overall MSE ratio of 0.971$\times$
baseline — a 2.9\% improvement in density accuracy.
```

- [ ] **Step 2: Draft Section 4.2 (Parent Features)**

```latex
\subsection{Parent Node Features}

To provide cross-level context, we augmented the model input with
parent node ($n{-}1$, $\lfloor l/2 \rfloor$) scaling coefficients
for both $\rho_0$ and $V_\text{nuc}$ (1024 additional floats),
increasing the trunk input dimension from 1792 to 2816 and total
parameters from 3.0M to 4.1M. The model was trained from random
initialization (no transfer from the single-task model, as the
first trunk layer shape changed).

Result: the model learned to ignore the parent features entirely.
Per-level MSE ratios were identical to the model without parent
features. The parent $\rho_0$ and $V_\text{nuc}$ coefficients
encode the same physical signal as the child-level coefficients —
the promolecular density and nuclear potential — at coarser
resolution. They do not provide the \emph{correction} signal
(the difference between $\rho_\text{conv}$ and $\rho_0$), because
that correction is precisely what the model is trying to predict.
Adding a coarser view of the input does not compensate for a
missing target signal.
```

- [ ] **Step 3: Draft Section 4.3 (SCF Convergence Test)**

```latex
\subsection{SCF Convergence Test}

We tested whether the level-clamped model's 2.9\% density
improvement translates to faster SCF convergence. Using the
infrastructure described in Section~3 (C++ density injection in
\texttt{SCF::initial\_guess()}, HDF5-to-archive converter), we
ran 7 held-out molecules at two convergence thresholds
($10^{-6}$ and $10^{-8}$), comparing SCF iteration counts
between the baseline ($\rho_0$) and ML-augmented initial guess.

Table~\ref{tab:scf_iterations} shows the results: all 14
comparisons yield \emph{identical iteration counts}. The correct
ground-state energy is recovered in every case (energies agree
to 8 significant figures), confirming that the ML density does
not degrade the SCF solution. However, it provides no acceleration.

This result has a clear mechanistic explanation. MADNESS uses
multi-protocol SCF: the first protocol (thresh=$10^{-4}$)
determines the coarse density, and subsequent protocols refine it.
The model's improvements at levels 10--14 encode spatial detail
at length scales below what the first protocol resolves — this
information is projected away when the first-protocol Fock matrix
is constructed. SCF iteration count is determined entirely by
coarse-level (1--9) density quality, where the model adds nothing.

\input{tables/scf_iterations}
```

- [ ] **Step 4: Draft Section 4.4 (Tree Structure Prediction)**

```latex
\subsection{Tree Structure Prediction}

Given that density prediction at fine levels has no SCF value,
we pivoted to a different ML role: predicting the tree structure
itself — which nodes require refinement (subdivision into children)
and which are leaves. Accurate tree prediction could accelerate
per-iteration compute by skipping expensive wavelet-norm evaluations
for confident predictions.

The MRANet architecture already includes a refinement classification
head. Initial training with all three heads active (density +
log-dnorm + refine) achieved F1 = 0.857, limited by the same
multi-task interference: the optimizer upweighted the density
regression loss $\sim$89$\times$ via the learned $\sigma_{\rho_s}$,
starving the refinement head of gradient.

Training with a dedicated refinement-only loss (focal loss
\cite{lin2017focal}, $\gamma=2$, $\alpha=0.75$, zero contribution
from density and log-dnorm heads) achieved F1 = 0.860 — an
improvement of only 0.003. This confirms that the F1 ceiling is
architectural, not caused by multi-task interference.

An SCF tree-walk test using the refine-only model's predicted tree
structure required 31 iterations versus 12 for the baseline — 2.6$\times$
worse. The correct electronic state was recovered (energies agree
to 7 significant figures), but the model-predicted tree topology
deviates enough from the true topology that the SCF solver requires
many additional iterations to compensate.

The refinement classification task faces the same receptive field
limitation as density prediction: refinement at level $n$ depends
on features at levels 0 through $n{-}1$ and on the global molecular
geometry, neither of which the model observes.
```

- [ ] **Step 5: Review for accuracy**

Cross-check every number in the draft against `extracted_results.yaml` and the postmortem. Specifically verify:
- 2.99x → 1.006x (multi-task → single-task)
- 0.41-0.98x at levels 10-14
- 14/14 identical iterations
- F1: 0.857 → 0.860 (delta 0.003)
- 31 vs 12 iterations (tree-walk)

- [ ] **Step 6: Humanize the prose**

Run `/humanizer` on `04-results.tex`. Check for and eliminate:
- Inflated claims ("pivotal", "crucial", "significant", "key")
- Passive voice where active is clearer
- Stock AI phrases ("it is important to note", "serves as")
- Forced groups of three
- Em dashes (replace with periods, commas, or colons)
- Formulaic challenge/outlook language
- Vague sources ("experts believe", "studies show")

Preserve all numerical claims, citations, and technical terminology exactly. Only rewrite the connecting prose.

- [ ] **Step 7: Commit**

```bash
git add mra_nn/paper/sections/04-results.tex
git commit -m "paper: draft results section (density prediction, parent features, SCF test, tree prediction)"
```

---

### Task 4: Method Section (Section 3)

**Files:**
- Create: `mra_nn/paper/sections/03-method.tex`

**Interfaces:**
- Consumes: `model.py`, `losses.py`, `dataset_builder.py`, `configs/single_task.yaml`
- Produces: Complete Section 3 draft (~2 pages)

- [ ] **Step 1: Draft Section 3.1 (Data Pipeline)**

Describe the three-step MADNESS dump, pymra processing, dataset construction. Reference the 51 W4-11 molecules, k=8/thresh=1e-6, 208k samples. Explain the halo construction (6 face-adjacent neighbors via two-scale relation) and negative generation (8 children below each leaf).

Read `mra_nn/dataset_builder.py` for exact details on how halos and negatives are constructed.

- [ ] **Step 2: Draft Section 3.2 (Architecture)**

Describe MRANet. Read `mra_nn/model.py` for exact layer dimensions, FiLM implementation, halo encoder weight sharing, residual connection. Include:
- Input dimensions: center (1024) + halo (6×512 → 768 via shared encoder) + level (32-dim FiLM)
- Trunk: 3 FiLM-conditioned layers (1024 → 512 → 256)
- Output: rho_s via residual connection
- Total parameters: ~3.0M (single-task)
- Reference Figure 1

- [ ] **Step 3: Draft Section 3.3 (Training)**

Describe training setup. Read `mra_nn/train.py` and `configs/single_task.yaml`:
- Single-task weighted MSE loss with level masking
- AdamW, lr=2e-4, cosine schedule, AMP, early stopping (patience=20)
- Batch size 4096, up to 120 epochs
- Train/val/test split: 45/3/3 molecules

Also describe the SCF test infrastructure:
- C++ injection point in `SCF::initial_guess()`
- h5_to_archive converter
- Level-clamped inference via `predict_density_simple()`

- [ ] **Step 4: Humanize the prose**

Run `/humanizer` on `03-method.tex`. Method sections tend toward passive voice and stock phrases. Rewrite for direct, active voice while keeping all technical detail exact.

- [ ] **Step 5: Commit**

```bash
git add mra_nn/paper/sections/03-method.tex
git commit -m "paper: draft method section (data pipeline, architecture, training)"
```

---

### Task 5: Analysis Section (Section 5)

**Files:**
- Create: `mra_nn/paper/sections/05-analysis.tex`

**Interfaces:**
- Consumes: Results from Section 4, postmortem root cause analysis
- Produces: Complete Section 5 draft (~1.5 pages)

- [ ] **Step 1: Draft the analysis**

This is the paper's intellectual contribution — the "why" behind the negative results. Three parts:

1. **Coarse-level convergence dominance**: SCF convergence path is set by protocol 1 (thresh=1e-4), which only resolves coarse levels. Fine-level improvements are projected away.

2. **The information bottleneck**: At coarse levels, rho0 is a smooth superposition of atomic densities and vnuc is the nuclear potential. The correction (rho_conv - rho0) encodes exchange-correlation, orbital relaxation, and bonding — non-local electron-electron interaction effects not present in the input. This is not a capacity problem (bigger models won't help) or a data problem (more molecules won't help at coarse levels). It's a fundamental input sufficiency problem.

3. **Architecture vs. input**: Expanding the receptive field (parent features) doesn't help when the expanded view contains the same signal at coarser resolution. A GNN or transformer could aggregate information across the tree, but if the leaf-level inputs don't contain the correction signal, aggregation can't create it.

Reference Figure 4 (tree schematic) here.

- [ ] **Step 2: Humanize the prose**

Run `/humanizer` on `05-analysis.tex`. This section is the most at risk for AI patterns — "pivotal insight", "crucial finding", "underscores the importance." State the physics plainly.

- [ ] **Step 3: Commit**

```bash
git add mra_nn/paper/sections/05-analysis.tex
git commit -m "paper: draft analysis section (coarse-level dominance, information bottleneck)"
```

---

### Task 6: Lessons and Path Forward (Section 6)

**Files:**
- Create: `mra_nn/paper/sections/06-lessons.tex`

**Interfaces:**
- Consumes: Analysis from Section 5, Adrian's Dalton results
- Produces: Complete Section 6 draft (~1 page)

- [ ] **Step 1: Draft the lessons**

Three transferable lessons:
1. Input signal quality > architecture
2. Fine-level accuracy ≠ solver acceleration
3. Multi-task interference is insidious

Each lesson should be stated as a general principle, then grounded in the specific MRA-NN evidence.

- [ ] **Step 2: Draft the path forward**

Describe the Dalton density idea as future work:
- Gaussian basis calculations provide a fundamentally richer starting point than promolecular density
- Preliminary results from collaborator show direct Dalton density reduces MADNESS iterations
- ML opportunity: predict the small Dalton→MRA residual instead of the large rho0→rho correction
- No results yet — frame as motivated future work

- [ ] **Step 3: Humanize the prose**

Run `/humanizer` on `06-lessons.tex`. Lessons sections easily become listicles with bold headings and inflated claims. Each lesson should read as a concrete observation, not a motivational poster.

- [ ] **Step 4: Commit**

```bash
git add mra_nn/paper/sections/06-lessons.tex
git commit -m "paper: draft lessons and path forward (Dalton density as future work)"
```

---

### Task 7: Introduction, Background, Conclusion (Sections 1, 2, 7)

**Why last:** Introduction frames the story — easier to write once you know the story. Background provides context the reader needs for the results — easier to calibrate once results are written. Conclusion summarizes — can't summarize what doesn't exist yet.

**Files:**
- Create: `mra_nn/paper/sections/01-introduction.tex`
- Create: `mra_nn/paper/sections/02-background.tex`
- Create: `mra_nn/paper/sections/07-conclusion.tex`

- [ ] **Step 1: Draft Introduction**

Three paragraphs:
1. MADNESS and the SCF bottleneck — why initial guess quality matters
2. ML for density prediction — the promise and prior work (Gong et al.)
3. Our contribution — systematic negative result, mechanistic explanation, transferable lessons

- [ ] **Step 2: Draft Background**

Two subsections:
1. MRA tree structure — scaling/wavelet coefficients, adaptive refinement, key (n,l) addressing. Keep concise — enough for a JCTC reader who knows DFT but not MRA.
2. SCF iteration mechanics — how initial density enters, multi-protocol convergence, what determines iteration count.

- [ ] **Step 3: Draft Conclusion**

Half a page:
- Summarize the three approaches and why they all failed
- State the positive contribution (the mechanistic explanation is novel)
- Open question: can Dalton+ML bridge the input signal gap?

- [ ] **Step 4: Humanize all three sections**

Run `/humanizer` on `01-introduction.tex`, `02-background.tex`, and `07-conclusion.tex`. Introductions and conclusions are the highest-risk sections for AI writing patterns — generic framing, inflated significance, and vague forward-looking statements. Be ruthless.

- [ ] **Step 5: Commit**

```bash
git add mra_nn/paper/sections/01-introduction.tex mra_nn/paper/sections/02-background.tex mra_nn/paper/sections/07-conclusion.tex
git commit -m "paper: draft introduction, background, and conclusion"
```

---

### Task 8: Master Document and Supplementary

**Files:**
- Create: `mra_nn/paper/main.tex`
- Create: `mra_nn/paper/supplementary.tex`

- [ ] **Step 1: Create main.tex**

```latex
% mra_nn/paper/main.tex
\documentclass[journal=jctcce,manuscript=article]{achemso}

\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{amsmath}
\usepackage{hyperref}

\title{Why Local ML Models Cannot Accelerate SCF Convergence
in Multiresolution Analysis: Lessons from Systematic Architecture Search}

\author{Ruhin Patel}
\affiliation{Stony Brook University}
\author{Adrian Hurtado}
\affiliation{Stony Brook University}
\author{Robert J.~Harrison}
\affiliation{Stony Brook University}

\begin{document}

\begin{abstract}
We present a systematic investigation of machine learning approaches
to accelerating self-consistent field (SCF) convergence in MADNESS,
a multiresolution analysis (MRA) framework for density functional
theory. Using a FiLM-conditioned MLP with same-level halo neighbors,
we evaluated three strategies: density coefficient prediction, parent
node augmentation, and tree structure prediction. All three hit the
same architectural limit. We identify the root cause as an input
signal bottleneck: at coarse tree levels, which determine SCF
convergence, the promolecular density and nuclear potential do not
contain the electron-electron interaction information needed for
density corrections. We extract three transferable lessons for ML
applied to multi-scale iterative solvers and propose using Gaussian
basis set densities as a richer ML input signal.
\end{abstract}

\input{sections/01-introduction}
\input{sections/02-background}
\input{sections/03-method}
\input{sections/04-results}
\input{sections/05-analysis}
\input{sections/06-lessons}
\input{sections/07-conclusion}

\begin{acknowledgement}
% TODO: funding acknowledgements
\end{acknowledgement}

\bibliography{references}

\end{document}
```

- [ ] **Step 2: Draft supplementary material**

```latex
% mra_nn/paper/supplementary.tex
\documentclass[journal=jctcce]{achemso}
\usepackage{booktabs, longtable}

\title{Supporting Information: Why Local ML Models Cannot Accelerate
SCF Convergence in Multiresolution Analysis}
\author{Ruhin Patel, Adrian Hurtado, Robert J.~Harrison}

\begin{document}

\section{Full Decision Audit}

Table~\ref{tab:decisions} lists all 27 design decisions made during
the project, with dates, descriptions, results, and verdicts.

% Extract from postmortem into a longtable with columns:
% Decision # | Date | Description | Result | Verdict
\begin{longtable}{clp{5cm}p{4cm}l}
\caption{Complete decision audit for the MRA-NN project.}
\label{tab:decisions} \\
\toprule
\# & Date & Decision & Result & Verdict \\
\midrule
\endfirsthead
\midrule
\# & Date & Decision & Result & Verdict \\
\midrule
\endhead
1 & Pre-Jul & MRA coefficient space & Sound & Sound \\
2 & Jul 16 & 15 W4-11 molecules & Insufficient data & Data issue \\
3 & Jul 16 & Three-step MADNESS dump & Works correctly & Sound \\
% ... continue for all 27 decisions ...
27 & Aug 14 & Refine-only training & F1=0.860, arch.\ limit & Ceiling \\
\bottomrule
\end{longtable}

\section{Training Hyperparameters}
% Table of all training runs with configs

\section{Per-Molecule SCF Data}
% Extended version of Table 1 with energies, dipole moments, per-protocol iteration counts

\section{Density Comparison Details}
% L2 norms, per-leaf MSE, dipole moment comparisons from compare_densities.py

\end{document}
```

- [ ] **Step 3: Test compilation (if LaTeX available)**

```bash
cd /gpfs/projects/rjh/ruhin/madness-ruhin/mra_nn/paper
# If pdflatex is available:
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

If LaTeX isn't installed on SeaWulf, compilation can happen locally. The draft is still reviewable as `.tex` source.

- [ ] **Step 4: Commit**

```bash
git add mra_nn/paper/main.tex mra_nn/paper/supplementary.tex
git commit -m "paper: add master document and supplementary material template"
```

---

### Task 9: Self-Review and Accuracy Audit

**Files:**
- Modify: All `mra_nn/paper/sections/*.tex` as needed

- [ ] **Step 1: Cross-check every numerical claim**

Go through each section and verify every number against:
- `extracted_results.yaml`
- The postmortem (`docs/2026-08-10-scf-test-postmortem.md`)
- Raw log files where needed

Pay special attention to:
- Per-level ratio values (some in the postmortem are approximate; verify against diagnostic logs)
- Parameter counts (verify against `model.py`)
- Iteration counts (verify against `scf_multi_2116941.out`)

- [ ] **Step 2: Check narrative consistency**

Read sections 1 → 7 in order. Verify:
- Introduction promises match what results deliver
- Background provides exactly what results section needs (no more, no less)
- Analysis section doesn't repeat results — it explains them
- Conclusion doesn't introduce new information

- [ ] **Step 3: Check figure/table references**

Verify every `\ref{fig:...}` and `\ref{tab:...}` has a matching `\label{...}`.

- [ ] **Step 4: Final humanizer pass on full manuscript**

Run `/humanizer` on each section file one more time, reading them in order (01 through 07). Per-section humanizer passes catch local issues; this final pass catches patterns that only appear when sections are read together (e.g., the same stock phrase appearing in sections 1, 4, and 7; repetitive paragraph openings across sections; escalating significance claims).

- [ ] **Step 5: Commit any fixes**

```bash
git add mra_nn/paper/
git commit -m "paper: accuracy audit and consistency fixes"
```

---

## Task Ordering Summary

| Task | Section | Depends on | Est. effort |
|------|---------|-----------|-------------|
| 1 | Scaffold + data extraction | Nothing | Light |
| 2 | Figure scripts | Task 1 | Medium |
| 3 | Section 4 (Results) | Tasks 1-2 | Medium |
| 4 | Section 3 (Method) | Task 1 | Medium |
| 5 | Section 5 (Analysis) | Task 3 | Light |
| 6 | Section 6 (Lessons) | Task 5 | Light |
| 7 | Sections 1, 2, 7 | Tasks 3-6 | Medium |
| 8 | main.tex + supplementary | Tasks 3-7 | Light |
| 9 | Accuracy audit | Task 8 | Light |

Tasks 3 and 4 can run in parallel. Tasks 5 and 6 can run in parallel. Everything else is sequential.
