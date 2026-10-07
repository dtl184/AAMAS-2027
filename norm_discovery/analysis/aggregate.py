"""Collect the saved raw/summary results into publication tables (results/tables/*.md, *.csv).

python -m analysis.aggregate
"""
from __future__ import annotations

import csv
import json
import os

from experiments.common import RESULTS


def _csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def _f(x, nd=2):
    if x in (None, "", "None"):
        return "–"
    try:
        return f"{float(x):.{nd}f}"
    except ValueError:
        return str(x)


def _ci(r, m, nd=2):
    if not r.get(f"{m}_mean"):
        return "–"
    return f"{_f(r[m + '_mean'], nd)} [{_f(r[m + '_ci_lo'], nd)}, {_f(r[m + '_ci_hi'], nd)}]"


def table_original():
    rows = _csv(os.path.join(RESULTS, "reconstruction", "original_results.csv"))
    L = ["| Domain | Method | Acc. | Viol. F1 | Unseen acc. | MAP hypothesis | P(intended) | Refinements |",
         "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['domain']} | {r['method']} | {_f(r['accuracy'])} | {_f(r['violation_f1'])} | "
                 f"{_f(r.get('unseen_accuracy'))} | {r.get('map_id') or '–'} | {_f(r.get('P_intended'), 3)} | "
                 f"{r.get('n_refinements') or '–'} |")
    return "\n".join(L)


def table_threshold():
    rows = _csv(os.path.join(RESULTS, "threshold_sensitivity", "summary.csv"))
    L = ["| Domain | Protocol | Baseline | ρ | Acc. [95% CI] | Viol. F1 | Unseen acc. | Refinement rate | Mean #ref. | P(intended) |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r["method"] != "full":
            continue
        rho = r["rho"] + (" **(paper)**" if r["rho"] == "0.1" else "")
        L.append(f"| {r['domain']} | {r['protocol']} | {r['baseline']} | {rho} | {_ci(r, 'accuracy')} | "
                 f"{_f(r.get('violation_f1_mean'))} | {_f(r.get('unseen_accuracy_mean'))} | {_f(r['refinement_rate'])} | "
                 f"{_f(r.get('n_refinements_mean'))} | {_f(r.get('P_intended_mean'), 3)} |")
    L += ["", "Reference (ρ-independent) on the same sequences:", "",
          "| Domain | Protocol | Method | Acc. [95% CI] | Unseen acc. | P(intended) |", "|---|---|---|---|---|---|"]
    for r in rows:
        if r["method"] in ("fixed", "oracle"):
            L.append(f"| {r['domain']} | {r['protocol']} | {r['method']} | {_ci(r, 'accuracy')} | "
                     f"{_f(r.get('unseen_accuracy_mean'))} | {_f(r.get('P_intended_mean'), 3)} |")
    return "\n".join(L)


def table_noise():
    rows = _csv(os.path.join(RESULTS, "noise_robustness", "summary.csv"))
    L = ["| Domain | Noise | q | Method | Acc. [95% CI] | Viol. F1 [95% CI] | Unseen acc. | P(intended) | MAP=intended | MAP≡intended (held-out) | Refinements |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['domain']} | {r['noise_type']} | {r['rate']} | {r['method']} | {_ci(r, 'accuracy')} | "
                 f"{_ci(r, 'violation_f1')} | {_f(r.get('unseen_accuracy_mean'))} | {_f(r.get('P_intended_mean'), 3)} | "
                 f"{_f(r.get('map_intended_rate'))} | {_f(r.get('map_equivalent_rate'))} | {_f(r.get('n_refinements_mean'))} |")
    return "\n".join(L)


def table_ident():
    d = os.path.join(RESULTS, "identifiability")
    rows = _csv(os.path.join(d, "prefix_posteriors.csv"))
    L = ["| Sequence | t | Demo context | P(H_int) | P(alt) | P(others) | Entropy | log BF int:alt | MAP | Acc. |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['sequence']} | {r['t']} | {r['demo_context'] or '(prior)'} | {_f(r['P_intended'], 3)} | "
                 f"{_f(r['P_alternative'], 3)} | {_f(r['P_others'], 3)} | {_f(r['entropy'])} | "
                 f"{_f(r['log_bayes_factor_int_vs_alt'])} | {r['map_id']} | {_f(r['accuracy'])} |")
    L += ["", "| Extra demo after D1–D5 | n | P(H_int) [95% CI] | log odds int:alt | Acc. | fraction MAP=H_int |",
          "|---|---|---|---|---|---|"]
    for r in _csv(os.path.join(d, "extra_demo_summary.csv")):
        L.append(f"| {r['selection']} | {r['n']} | {_ci(r, 'P_intended', 3)} | {_f(r['log_posterior_odds_int_vs_alt_mean'])} | "
                 f"{_f(r['accuracy_mean'])} | {_f(r['fraction_MAP_intended'])} |")
    L += ["", "| Competitor | P after D1–D5 | Most diagnostic context | MI (nats, equal pair prior) | E|log BF| | Best ordinary context MI |",
          "|---|---|---|---|---|---|"]
    for r in _csv(os.path.join(d, "mi_landscape.csv")):
        L.append(f"| {r['competitor']} | {_f(r['competitor_posterior_after_D5'], 3)} | {r['best_context']} | "
                 f"{_f(r['best_MI'], 3)} | {_f(r['best_expected_abs_logBF'], 1)} | {_f(r['best_standard_MI'], 3)} |")
    return "\n".join(L)


def table_dl():
    p = os.path.join(RESULTS, "supplement_dl_coding", "summary.csv")
    if not os.path.exists(p):
        return "(not run)"
    L = ["| Coding | Domain | Protocol | Method | ρ / baseline | Acc. | Unseen acc. | P(intended) | MAP counts |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in _csv(p):
        L.append(f"| {r['dl_coding']} | {r['domain']} | {r['protocol']} | {r['method']} | {r['rho'] or '–'} / {r['baseline']} | "
                 f"{_ci(r, 'accuracy')} | {_f(r.get('unseen_accuracy_mean'))} | {_f(r['P_intended_mean'], 3)} | {r['map_counts']} |")
    return "\n".join(L)


def main():
    out = os.path.join(RESULTS, "tables")
    os.makedirs(out, exist_ok=True)
    parts = {"original_protocol": table_original, "threshold_sensitivity": table_threshold,
             "noise_robustness": table_noise, "identifiability": table_ident, "dl_coding_supplement": table_dl}
    allmd = []
    for name, fn in parts.items():
        try:
            md = fn()
        except FileNotFoundError as e:
            md = f"(missing input: {e.filename})"
        with open(os.path.join(out, f"{name}.md"), "w") as f:
            f.write(md + "\n")
        allmd += [f"## {name}", "", md, ""]
    with open(os.path.join(out, "ALL_TABLES.md"), "w") as f:
        f.write("\n".join(allmd))
    print(f"wrote tables to {out}")


if __name__ == "__main__":
    main()
