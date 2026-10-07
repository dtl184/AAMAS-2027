"""Re-run every experiment and rebuild all tables/figures.

python -m experiments.run_all [--quick] [--set key=value ...]

Order: reconstruction + sanity checks -> threshold sensitivity -> noise robustness -> identifiability
-> description-length-coding supplement -> aggregation (tables + RESULTS tables).
Any CLI flags are forwarded unchanged to every experiment.
"""
from __future__ import annotations

import subprocess
import sys
import time

STEPS = ["experiments.reconstruct_original", "experiments.threshold_sensitivity", "experiments.noise_robustness",
         "experiments.identifiability", "experiments.supplement_dl_coding", "analysis.aggregate"]


def main():
    args = sys.argv[1:]
    for mod in STEPS:
        t0 = time.time()
        fwd = [] if mod == "analysis.aggregate" else args
        print(f"=== {mod} {' '.join(fwd)}", flush=True)
        r = subprocess.run([sys.executable, "-m", mod, *fwd])
        if r.returncode:
            sys.exit(f"{mod} failed with exit code {r.returncode}")
        print(f"=== {mod} finished in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
