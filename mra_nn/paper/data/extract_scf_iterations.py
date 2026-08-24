"""Extract SCF iteration counts from multi-molecule test log.

Parses scf_multi_2116941.out to verify the iteration counts
in extracted_results.yaml. Run from repo root.
"""
import re
from pathlib import Path

LOG_PATH = "/gpfs/projects/rjh/ruhin/mra_nn/logs/scf_multi_2116941.out"


def extract_iterations(log_text: str) -> list[dict]:
    """Parse the multi-molecule SCF test log.

    Each molecule block has a header like:
        ### ch3oh @ thresh=1e-6  (18 electrons) ###
    followed by [3/5] Baseline SCF and [4/5] ML-guess SCF sections.
    We count 'Iteration N' lines per section (both protocols combined).
    """
    pattern = r"### (\S+) @ thresh=(\S+)\s+\((\d+) electrons\)"
    headers = list(re.finditer(pattern, log_text))

    results = []
    for idx, m in enumerate(headers):
        mol = m.group(1)
        thresh = m.group(2)
        ne = int(m.group(3))

        start = m.end()
        end = headers[idx + 1].start() if idx + 1 < len(headers) else len(log_text)
        block = log_text[start:end]

        # Find baseline and ML sections
        baseline_match = re.search(
            r"\[3/5\].*?Baseline(.*?)(?:\[4/5\]|\[5/5\])", block, re.DOTALL
        )
        ml_match = re.search(
            r"\[4/5\].*?ML-guess(.*?)(?:\[5/5\])", block, re.DOTALL
        )

        baseline_iters = (
            len(re.findall(r"Iteration \d+", baseline_match.group(1)))
            if baseline_match
            else 0
        )
        ml_iters = (
            len(re.findall(r"Iteration \d+", ml_match.group(1)))
            if ml_match
            else 0
        )

        # Extract energies from summary line
        energy_match = re.search(
            r"Baseline:\s+\d+ iter.*?energy=\s*([-\d.]+).*?"
            r"ML-guess:\s+\d+ iter.*?energy=\s*([-\d.]+)",
            block,
            re.DOTALL,
        )
        baseline_energy = float(energy_match.group(1)) if energy_match else None
        ml_energy = float(energy_match.group(2)) if energy_match else None

        results.append(
            {
                "molecule": mol,
                "threshold": thresh,
                "n_electrons": ne,
                "baseline_iters": baseline_iters,
                "ml_iters": ml_iters,
                "baseline_energy": baseline_energy,
                "ml_energy": ml_energy,
            }
        )
    return results


if __name__ == "__main__":
    log_text = Path(LOG_PATH).read_text()
    results = extract_iterations(log_text)
    print(f"Extracted {len(results)} test results")
    all_same = True
    for r in results:
        same = "SAME" if r["baseline_iters"] == r["ml_iters"] else "DIFF"
        if same == "DIFF":
            all_same = False
        print(
            f"  {r['molecule']:12s} @ {r['threshold']}: "
            f"baseline={r['baseline_iters']:2d}  ML={r['ml_iters']:2d}  [{same}]  "
            f"E_base={r['baseline_energy']:.8f}  E_ml={r['ml_energy']:.8f}"
        )
    print(f"\nAll identical: {all_same} ({len(results)}/{len(results)})")
