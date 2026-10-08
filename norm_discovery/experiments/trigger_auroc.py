"""AUROC of the four refinement-trigger statistics for detecting useful refinement opportunities.

Inputs (results/gpt55/refinement_trigger_comparison/):
  paths.jsonl            unrefined learner path per (domain, protocol, run): all four statistics at every t
  labels_complete.jsonl  counterfactual GPT-5.5 refinement labels (falls back to labels.jsonl); only successful records
An opportunity is (domain, protocol, run_index, t), t >= 3.  Positive class = "useful": refining with GPT-5.5 at t
raises P(intended-equivalent) by >= 0.05 or raises held-out / unseen accuracy (definition fixed in the labelling
stage; evaluation only).  A trigger fires on LOW scores, so the AUROC of a statistic s is P(s(useful) < s(not useful)),
ties counted 1/2.  Two forms are scored for each family:
  value    s_t itself
  margin   s_t - min_{2 <= i < t} s_i  (distance to the running minimum of earlier scores on the same unrefined path,
           i.e. the quantity the running-minimum trigger thresholds)
Uncertainty: 95% bootstrap CI resampling RUNS (opportunities within a run are dependent), 4000 replicates.
Reportability rule (fixed before looking at AUROC values): a cell is reportable only if its labels are complete
(no failed opportunities) and it has >= 10 positives and >= 10 negatives; otherwise descriptive only.

    /home/train/anaconda3/bin/python -m experiments.gpt55_trigger_comparison auroc
"""
from __future__ import annotations

import collections
import json
import os

import numpy as np
from scipy.stats import spearmanr

from analysis import plotting as P
from experiments.common import RESULTS, ROOT, write_csv, write_json
from inference.triggers import FAMILIES, FAMILY_NAMES

D = os.path.join(RESULTS, "gpt55", "refinement_trigger_comparison")
OUT = os.path.join(D, "auroc")
FIG = os.path.join(ROOT, "figures", "gpt55", "refinement_trigger_comparison")
MIN_PER_CLASS = 10
N_BOOT = 4000


def auroc(scores, labels):
    """P(score of a positive < score of a negative); positives = useful (trigger fires on low scores)."""
    s, y = np.asarray(scores, float), np.asarray(labels, bool)
    pos, neg = s[y], s[~y]
    if len(pos) == 0 or len(neg) == 0:
        return None
    less = (pos[:, None] < neg[None, :]).sum()
    ties = (pos[:, None] == neg[None, :]).sum()
    return float((less + 0.5 * ties) / (len(pos) * len(neg)))


def load():
    with open(os.path.join(D, "paths.jsonl")) as f:
        paths = [json.loads(l) for l in f]
    lp = os.path.join(D, "labels_complete.jsonl")
    src = lp if os.path.exists(lp) else os.path.join(D, "labels.jsonl")
    with open(src) as f:
        labels = [json.loads(l) for l in f]
    return paths, labels, src


def opportunities(paths, labels):
    lab = {(l["domain"], l["protocol"], l["run_index"], l["t"]): l for l in labels if not l.get("error")}
    failed = {(l["domain"], l["protocol"], l["run_index"], l["t"]) for l in labels if l.get("error")}
    rows = []
    for p in paths:
        if p.get("error"):
            continue
        steps = sorted(p["steps"], key=lambda s: s["t"])
        for i, s in enumerate(steps):
            if s["t"] < 3:
                continue
            key = (p["domain"], p["protocol"], p["run_index"], s["t"])
            prev = steps[:i]
            row = {"domain": key[0], "protocol": key[1], "run_index": key[2], "t": key[3],
                   "demo_len": s["demo_len"], "C_task": s["C_task"], "labelled": key in lab, "failed": key in failed,
                   "useful": lab[key]["useful"] if key in lab else None}
            for f in FAMILIES:
                v = s[f"score_{f}"]
                row[f"value_{f}"] = v
                row[f"margin_{f}"] = v - min(x[f"score_{f}"] for x in prev) if prev else None
            rows.append(row)
    return rows


def cluster_boot(rows, col, rng):
    runs = sorted({r["run_index"] for r in rows})
    by = collections.defaultdict(list)
    for r in rows:
        by[r["run_index"]].append(r)
    vals = []
    for _ in range(N_BOOT):
        pick = rng.choice(runs, size=len(runs), replace=True)
        rs = [r for k in pick for r in by[k]]
        a = auroc([r[col] for r in rs], [r["useful"] for r in rs])
        if a is not None:
            vals.append(a)
    if len(vals) < N_BOOT * 0.5:
        return None, None
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def main(cfg=None):
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    paths, labels, src = load()
    opp = opportunities(paths, labels)
    rng = np.random.default_rng(20261008)
    cells = [("cart", "pool"), ("aisle", "pool"), ("cart", "orig"), ("aisle", "orig"), ("both", "pool")]
    table, status = [], []
    for dom, proto in cells:
        allr = [r for r in opp if (dom == "both" or r["domain"] == dom) and r["protocol"] == proto]
        rs = [r for r in allr if r["labelled"]]
        npos, nneg = sum(r["useful"] for r in rs), sum(not r["useful"] for r in rs)
        n_fail = sum(r["failed"] for r in allr)
        complete = n_fail == 0 and len(rs) == len(allr)
        reportable = complete and npos >= MIN_PER_CLASS and nneg >= MIN_PER_CLASS
        reason = ("ok" if reportable else
                  "; ".join(x for x in [None if complete else f"{n_fail} failed / {len(allr) - len(rs)} unlabelled opportunities",
                                        None if npos >= MIN_PER_CLASS else f"only {npos} useful",
                                        None if nneg >= MIN_PER_CLASS else f"only {nneg} not-useful"] if x))
        status.append({"domain": dom, "protocol": proto, "n_opportunities": len(allr), "n_labelled": len(rs),
                       "n_useful": npos, "n_not_useful": nneg, "n_failed": n_fail, "labels_complete": complete,
                       "reportable": reportable, "reason": reason})
        if dom == "both":
            note = "pooled across domains (tests a single scale; mixes domain-level differences)"
        else:
            note = ""
        for f in FAMILIES:
            for form in ("value", "margin"):
                col = f"{form}_{f}"
                a = auroc([r[col] for r in rs], [r["useful"] for r in rs]) if rs else None
                lo, hi = cluster_boot(rs, col, rng) if (a is not None and len({r['run_index'] for r in rs}) > 2) else (None, None)
                rho_len = (float(spearmanr([r["demo_len"] for r in rs], [r[col] for r in rs]).correlation)
                           if len(rs) > 3 else None)
                table.append({"domain": dom, "protocol": proto, "family": f, "family_name": FAMILY_NAMES[f], "form": form,
                              "AUROC": a, "CI_lo": lo, "CI_hi": hi, "n_useful": npos, "n_not_useful": nneg,
                              "reportable": reportable, "spearman_with_demo_length": rho_len, "note": note})
    write_csv(os.path.join(OUT, "auroc.csv"), table)
    write_csv(os.path.join(OUT, "cell_status.csv"), status)
    write_csv(os.path.join(OUT, "opportunities.csv"), opp)
    write_json(os.path.join(OUT, "auroc.json"), {"label_source": src, "min_per_class": MIN_PER_CLASS,
                                                 "bootstrap": f"{N_BOOT} run-clustered replicates", "cells": status,
                                                 "auroc": table})
    roc_figure(opp)
    for s in status:
        print(json.dumps(s))
    for t in table:
        if t["AUROC"] is not None:
            ci = "" if t["CI_lo"] is None else f" [{t['CI_lo']:.2f}, {t['CI_hi']:.2f}]"
            print(f"{t['domain']:5s} {t['protocol']:4s} {t['family']:4s} {t['form']:6s} AUROC={t['AUROC']:.3f}{ci} "
                  f"(+{t['n_useful']}/-{t['n_not_useful']}) reportable={t['reportable']} rho_len={t['spearman_with_demo_length']}")


def roc_figure(opp):
    P.setup()
    fig, axes = P.plt.subplots(1, 2, figsize=(10, 4))
    for ax, dom in zip(axes, ("cart", "aisle")):
        rs = [r for r in opp if r["domain"] == dom and r["protocol"] == "pool" and r["labelled"]]
        y = np.array([r["useful"] for r in rs], bool)
        for i, f in enumerate(FAMILIES):
            s = np.array([r[f"margin_{f}"] for r in rs], float)
            if y.all() or (~y).all():
                continue
            thr = np.unique(s)
            tpr = [0.0] + [float((s[y] <= t).mean()) for t in thr]
            fpr = [0.0] + [float((s[~y] <= t).mean()) for t in thr]
            ax.plot(fpr, tpr, color=P.SLOTS[i], lw=1.8, label=f"{FAMILY_NAMES[f]} (AUROC {auroc(s, y):.2f})")
        ax.plot([0, 1], [0, 1], color=P.INK2, ls=":", lw=0.8)
        ax.set_xlabel("false-positive rate (not-useful opportunities flagged)")
        ax.set_ylabel("true-positive rate (useful opportunities flagged)")
        ax.set_title(f"{'Exp. 1: cart-use' if dom == 'cart' else 'Exp. 2: aisle-use'} – 20-demo pools, margin form\n"
                     f"(+{int(y.sum())} useful / −{int((~y).sum())} not useful)", fontsize=9)
        ax.legend(fontsize=7, loc="lower right")
    P.save(fig, os.path.join(FIG, "auroc_roc_curves.png"))


if __name__ == "__main__":
    main()
