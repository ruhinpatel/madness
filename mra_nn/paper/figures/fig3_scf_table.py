"""Generate LaTeX table of multi-molecule SCF iteration comparisons.

Data source: scf_multi_2116941.out (verified by extract_scf_iterations.py)
14/14 identical iteration counts.
"""
from pathlib import Path

# Verified data from scf_multi_2116941.out
# Format: (molecule_display, n_electrons, thresh_display, baseline, ml, baseline_E, ml_E)
data = [
    (r"CH$_3$OH",    18, r"$10^{-6}$", 12, 12, -114.85038034, -114.85038032),
    (r"CH$_3$OH",    18, r"$10^{-8}$", 12, 12, -114.85038034, -114.85038032),
    ("Ethanol",      26, r"$10^{-6}$", 12, 12, -153.81559584, -153.81559579),
    ("Ethanol",      26, r"$10^{-8}$", 12, 12, -153.81559584, -153.81559579),
    (r"SO$_2$",      32, r"$10^{-6}$", 14, 14, -546.34469275, -546.34469263),
    (r"SO$_2$",      32, r"$10^{-8}$", 14, 14, -546.34469265, -546.34469262),
    (r"HN$_3$",      22, r"$10^{-6}$", 13, 13, -163.57199524, -163.57199523),
    (r"HN$_3$",      22, r"$10^{-8}$", 13, 13, -163.57199524, -163.57199523),
    (r"H$_2$O$_2$",  18, r"$10^{-6}$", 12, 12, -150.55376660, -150.55376665),
    (r"H$_2$O$_2$",  18, r"$10^{-8}$", 12, 12, -150.55376660, -150.55376665),
    (r"C$_2$H$_2$",  14, r"$10^{-6}$", 10, 10,  -76.63064903,  -76.63064903),
    (r"C$_2$H$_2$",  14, r"$10^{-8}$", 10, 10,  -76.63064903,  -76.63064903),
    ("Glyoxal",      30, r"$10^{-6}$", 13, 13, -226.15979738, -226.15979728),
    ("Glyoxal",      30, r"$10^{-8}$", 13, 13, -226.15979738, -226.15979728),
]

lines = [
    r"\begin{table}[htbp]",
    r"\centering",
    r"\caption{SCF iteration counts for level-clamped ML density (levels 10--14)"
    r" vs.\ promolecular baseline ($\rho_0$). All 14 comparisons yield identical"
    r" iteration counts. Ground-state energies agree to 8+ significant figures.}",
    r"\label{tab:scf_iterations}",
    r"\begin{tabular}{lccrrc}",
    r"\toprule",
    r"Molecule & $N_e$ & Thresh & Baseline & ML & $E_\text{base}$ (Ha) \\",
    r"\midrule",
]
for mol, ne, thresh, base, ml, e_base, e_ml in data:
    lines.append(
        f"{mol} & {ne} & {thresh} & {base} & {ml} & {e_base:.8f} \\\\"
    )
lines += [
    r"\bottomrule",
    r"\end{tabular}",
    r"\end{table}",
]

out = Path("/gpfs/projects/rjh/ruhin/madness-ruhin/mra_nn/paper/tables/scf_iterations.tex")
out.write_text("\n".join(lines) + "\n")
print(f"Written to {out}")
