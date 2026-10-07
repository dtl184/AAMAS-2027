"""Supplementary sensitivity analysis: description-length coding.

Added AFTER observing the primary results (see README): under the primary "nodes" coding a coordinate atom
at(x,y) costs 1 unit while numeric constants inside comparisons cost 1 unit each, so a disjunction of
memorised cells is cheap relative to a structural definition.  This script re-runs the threshold-pool
protocol (same datasets/seeds) and the original 3-demonstration protocol with the token-consistent coding
("tokens": at(x,y) = predicate + 2 constants = 3 units).  It does not replace the primary results.

python -m experiments.supplement_dl_coding [--quick]
"""
from __future__ import annotations

import os
import time

import numpy as np

from experiments.common import (bootstrap_ci, out_dir, parse_args, pmap, run_metadata, write_csv, write_json,
                                write_jsonl)
from experiments.threshold_sensitivity import _job

CONDITIONS = [("fixed", None, "min"), ("full", 0.1, "min"), ("full", 0.4, "min"), ("full", 0.8, "median"),
              ("oracle", None, "min")]


def main():
    cfg = parse_args("Description-length coding sensitivity")
    od = out_dir(cfg, "supplement_dl_coding")
    write_json(os.path.join(od, "metadata.json"), run_metadata("supplement_dl_coding", cfg))
    t0 = time.time()
    jobs = []
    for coding in ("nodes", "tokens"):
        c = dict(cfg, dl_coding=coding)
        for key in cfg["domains"]:
            for m, rho, b in CONDITIONS:
                jobs.append((key, "original", (0, 1, 2), m, rho, b, c))
                for s in range(cfg["threshold_pool_seeds"]):
                    jobs.append((key, "pool", s, m, rho, b, c))
    res = pmap(_job_wrap, jobs, cfg["workers"])
    write_jsonl(os.path.join(od, "runs.jsonl"), res)
    summ = []
    for coding in ("nodes", "tokens"):
        for key in cfg["domains"]:
            for proto in ("original", "pool"):
                for m, rho, b in CONDITIONS:
                    rs = [r for r in res if r["dl_coding"] == coding and r["domain"] == key and r["protocol"] == proto
                          and r["method"] == m and r["rho"] == rho and r["baseline"] == b]
                    row = {"dl_coding": coding, "domain": key, "protocol": proto, "method": m, "rho": rho,
                           "baseline": b, "n": len(rs)}
                    for mt in ("accuracy", "violation_f1", "unseen_accuracy"):
                        v = [r[mt] for r in rs if r.get(mt) is not None]
                        if v:
                            mu, lo, hi = bootstrap_ci(v, cfg["bootstrap"])
                            row.update({f"{mt}_mean": mu, f"{mt}_ci_lo": lo, f"{mt}_ci_hi": hi})
                    row["P_intended_mean"] = float(np.mean([r.get("P_intended") or 0.0 for r in rs]))
                    row["map_intended_rate"] = float(np.mean([bool(r.get("map_is_intended")) for r in rs]))
                    row["refinement_rate"] = float(np.mean([(r.get("n_refinements") or 0) > 0 for r in rs]))
                    maps = {}
                    for r in rs:
                        maps[r["map_id"]] = maps.get(r["map_id"], 0) + 1
                    row["map_counts"] = maps
                    summ.append(row)
    write_csv(os.path.join(od, "summary.csv"), summ)
    for r in summ:
        print(f"{r['dl_coding']:6s} {r['domain']:5s} {r['protocol']:8s} {r['method']:6s} rho={r['rho']} {r['baseline']:6s} "
              f"acc={r.get('accuracy_mean', float('nan')):.3f} unseen={r.get('unseen_accuracy_mean', float('nan')):.3f} "
              f"Pint={r['P_intended_mean']:.3f} maps={r['map_counts']}")
    print(f"done in {time.time() - t0:.0f}s")


def _job_wrap(job):
    r = _job(job)
    r["dl_coding"] = job[-1]["dl_coding"]
    return r


if __name__ == "__main__":
    main()
