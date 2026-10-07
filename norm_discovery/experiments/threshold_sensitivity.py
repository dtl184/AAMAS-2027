"""NEW EXPERIMENT 1: sensitivity of abstraction discovery to the refinement threshold rho.

Protocol A (original): the three original demonstrations, in all 3! = 6 orders (the canonical order
D1,D2,D3 is the paper's protocol), full method with rho in rho_grid, running-minimum baseline
(primary) and median baseline (secondary diagnostic).
Protocol B (pools): `threshold_pool_seeds` independently generated clean training sequences of
`threshold_pool_demos` demonstrations (random tasks from the task pool, optimal paths with random
tie-breaking); same grid.  Fixed- and oracle-abstraction runs on the same sequences are reference lines.

python -m experiments.threshold_sensitivity [--quick] [--set rho_grid=[...] threshold_pool_seeds=20]
"""
from __future__ import annotations

import itertools
import os
import time

import numpy as np
from scipy.stats import spearmanr

from analysis import plotting as P
from environment.domains import get_spec
from experiments.common import (FIGURES, bootstrap_ci, flat, make_pool_dataset, out_dir, parse_args, pmap,
                                read_jsonl, run_metadata, run_method, seed_for, write_csv, write_json, write_jsonl)

BASELINES = ["min", "median"]


def _job(job):
    key, protocol, idx, method, rho, baseline, cfg = job
    spec = get_spec(key)
    if protocol == "original":
        demos = [spec.training[i] for i in idx]
    else:
        demos = make_pool_dataset(spec, cfg["threshold_pool_demos"], seed_for(cfg, "pool", key, idx))
    r = run_method(spec, method, demos, cfg, rho=rho, baseline=baseline)
    trace = [{k: s.get(k) for k in ("t", "demo", "demo_len", "log_p", "log_b", "log_ratio_p_over_b", "triggered",
                                     "n_hypotheses", "P_intended", "map_id")} for s in r["trace"]]
    return {"domain": key, "protocol": protocol, "order_or_seed": list(idx) if protocol == "original" else idx,
            "method": method, "rho": rho, "baseline": baseline, **flat(r),
            "first_refinement": (r["refinement_points"][0] if r.get("refinement_points") else None),
            "trace": trace}


def build_jobs(cfg):
    jobs = []
    for key in cfg["domains"]:
        for order in itertools.permutations(range(3)):
            for rho in cfg["rho_grid"]:
                for b in BASELINES:
                    jobs.append((key, "original", order, "full", rho, b, cfg))
            for m in ("fixed", "oracle"):
                jobs.append((key, "original", order, m, None, "min", cfg))
        for s in range(cfg["threshold_pool_seeds"]):
            for rho in cfg["rho_grid"]:
                for b in BASELINES:
                    jobs.append((key, "pool", s, "full", rho, b, cfg))
            for m in ("fixed", "oracle"):
                jobs.append((key, "pool", s, m, None, "min", cfg))
    return jobs


def summarise(rows, cfg):
    out = []
    keys = sorted({(r["domain"], r["protocol"], r["method"], r["rho"], r["baseline"]) for r in rows},
                  key=lambda k: (k[0], k[1], k[2], -1 if k[3] is None else k[3], k[4]))
    for (d, p, m, rho, b) in keys:
        rs = [r for r in rows if (r["domain"], r["protocol"], r["method"], r["rho"], r["baseline"]) == (d, p, m, rho, b)]
        row = {"domain": d, "protocol": p, "method": m, "rho": rho, "baseline": b, "n_runs": len(rs),
               "is_paper_rho": rho == 0.1}
        for metric in ("accuracy", "violation_f1", "unseen_accuracy", "P_intended", "n_refinements", "n_hypotheses",
                       "runtime_s"):
            if metric == "P_intended" and m != "mlci":   # intended absent from H => posterior mass 0
                vals = [r.get(metric) or 0.0 for r in rs]
            else:
                vals = [r[metric] for r in rs if r.get(metric) is not None]
            if vals:
                mu, lo, hi = bootstrap_ci(vals, cfg["bootstrap"])
                row.update({f"{metric}_mean": mu, f"{metric}_ci_lo": lo, f"{metric}_ci_hi": hi,
                            f"{metric}_sd": float(np.std(vals))})
        row["refinement_rate"] = float(np.mean([(r.get("n_refinements") or 0) > 0 for r in rs]))
        row["map_intended_rate"] = float(np.mean([bool(r.get("map_is_intended")) for r in rs]))
        row["map_equivalent_rate"] = float(np.mean([bool(r.get("map_equivalent_on_heldout")) for r in rs]))
        fr = [r["first_refinement"] for r in rs if r.get("first_refinement")]
        row["first_refinement_median"] = float(np.median(fr)) if fr else None
        row["first_refinement_values"] = sorted(fr)
        out.append(row)
    return out


def running_min_diagnostics(rows):
    """How much of the variation in log p_t is explained by demonstration length (task cost)?"""
    out = {}
    for d in sorted({r["domain"] for r in rows}):
        lp, ln, ratios_trig, lens_trig, lens_all = [], [], [], [], []
        for r in rows:
            if r["domain"] != d or r["protocol"] != "pool":
                continue
            if r["method"] == "fixed":
                for s in r["trace"]:
                    if s["log_p"] is not None:
                        lp.append(s["log_p"]); ln.append(s["demo_len"])
            if r["method"] == "full" and r["rho"] == 0.1 and r["baseline"] == "min":
                med = np.median([s["demo_len"] for s in r["trace"]])
                for s in r["trace"]:
                    lens_all.append(s["demo_len"] > med)
                    if s["triggered"]:
                        lens_trig.append(s["demo_len"] > med)
        if lp:
            rho_s = spearmanr(ln, lp).correlation
            A = np.vstack([ln, np.ones(len(ln))]).T
            coef, res, *_ = np.linalg.lstsq(A, lp, rcond=None)
            pred = A @ coef
            r2 = 1 - np.sum((np.array(lp) - pred) ** 2) / np.sum((np.array(lp) - np.mean(lp)) ** 2)
            out[d] = {"n_predictive_values": len(lp), "spearman_logp_vs_length": float(rho_s),
                      "slope_nats_per_step": float(coef[0]), "R2_length_only": float(r2),
                      "fraction_triggers_on_longer_than_median_demos(rho=0.1,min)":
                          (float(np.mean(lens_trig)) if lens_trig else None), "n_triggers": len(lens_trig)}
    return out


def plots(summary, rows, cfg, fig_dir):
    P.setup()
    rhos = cfg["rho_grid"]
    paths = []
    domains = cfg["domains"]

    def get(d, p, m, b, metric):
        xs, mu, lo, hi = [], [], [], []
        for rho in rhos:
            row = next((s for s in summary if s["domain"] == d and s["protocol"] == p and s["method"] == m
                        and s["rho"] == rho and s["baseline"] == b), None)
            if row and f"{metric}_mean" in row:
                xs.append(rho); mu.append(row[f"{metric}_mean"]); lo.append(row[f"{metric}_ci_lo"]); hi.append(row[f"{metric}_ci_hi"])
        return np.array(xs), np.array(mu), np.array(lo), np.array(hi)

    def ref(d, p, m, metric):
        row = next((s for s in summary if s["domain"] == d and s["protocol"] == p and s["method"] == m), None)
        return None if row is None or f"{metric}_mean" not in row else row[f"{metric}_mean"]

    specs = [("accuracy", "Held-out accuracy", "threshold_accuracy.png"),
             ("violation_f1", "Violation F1", "threshold_f1.png"),
             ("n_refinements", "Refinements per run", "threshold_refinements.png"),
             ("P_intended", "Posterior of intended hypothesis", "threshold_p_intended.png"),
             ("unseen_accuracy", "Unseen-aisle accuracy", "threshold_unseen_accuracy.png")]
    for metric, ylabel, fname in specs:
        doms = [d for d in domains if not (metric == "unseen_accuracy" and d == "cart")]
        fig, axes = P.plt.subplots(1, len(doms), figsize=(4.6 * len(doms), 3.4), squeeze=False)
        for ax, d in zip(axes[0], doms):
            for p, b, ls, lab in (("pool", "min", "-", "pool, running min (primary)"),
                                  ("pool", "median", "--", "pool, median baseline"),
                                  ("original", "min", ":", "original 3 demos (6 orders), running min")):
                x, mu, lo, hi = get(d, p, "full", b, metric)
                if len(x) == 0:
                    continue
                st = dict(P.METHOD_STYLE["full"])
                ax.plot(x, mu, ls=ls, color=st["color"] if p == "pool" else P.SLOTS[6], marker=st["marker"] if b == "min" else "x",
                        label=lab, ms=4)
                if p == "pool" and b == "min":
                    ax.fill_between(x, lo, hi, color=st["color"], alpha=0.15, lw=0)
            if metric not in ("n_refinements",):
                for m in ("fixed", "oracle"):
                    v = ref(d, "pool", m, metric)
                    if v is not None:
                        ax.axhline(v, color=P.METHOD_STYLE[m]["color"], lw=1.2, ls="-.",
                                   label=f"{P.METHOD_STYLE[m]['label']} (pool)")
            ax.axvline(0.1, color=P.INK2, lw=0.8, ls=":")
            ax.text(0.1, ax.get_ylim()[1], " ρ=0.1 (paper)", fontsize=7, color=P.INK2, va="top")
            ax.set_xscale("log")
            ax.set_xlabel("refinement threshold ρ")
            ax.set_ylabel(ylabel)
            ax.set_title({"cart": "Exp. 1: cart-use", "aisle": "Exp. 2: aisle-use"}[d])
        axes[0][-1].legend(fontsize=6.5, loc="best")
        p_ = os.path.join(fig_dir, fname)
        P.save(fig, p_)
        paths.append(p_)
    # ---- representative traces: log p_t and log(rho * b_t) over the demonstration index
    fig, axes = P.plt.subplots(2, len(domains), figsize=(4.8 * len(domains), 6.2), squeeze=False)
    for j, d in enumerate(domains):
        for i, (p, idx) in enumerate((("original", [0, 1, 2]), ("pool", 0))):
            ax = axes[i][j]
            r = next((r for r in rows if r["domain"] == d and r["protocol"] == p and r["method"] == "fixed"
                      and r["order_or_seed"] == idx), None)
            if r is None:
                continue
            ts = [s["t"] for s in r["trace"] if s["log_p"] is not None]
            lp = [s["log_p"] for s in r["trace"] if s["log_p"] is not None]
            ax.plot(ts, lp, color=P.INK, marker="o", ms=4, label="log p_t (fixed-abstraction hypotheses)")
            # running-minimum and median baselines of the non-refining sequence
            bmin = [None] + list(np.minimum.accumulate(lp))[:-1]
            bmed = [None] + [float(np.median(lp[:k])) for k in range(1, len(lp))]
            for rho, col in ((0.01, P.SLOTS[2]), (0.1, P.SLOTS[0]), (0.8, P.SLOTS[1])):
                yy = [None if b is None else b + np.log(rho) for b in bmin]
                ax.plot(ts[1:], yy[1:], color=col, lw=1.4, drawstyle="steps-post", marker="_", ms=14, mew=2,
                        label=f"log(ρ·b_t), min, ρ={rho}")
            yy = [None if b is None else b + np.log(0.1) for b in bmed]
            ax.plot(ts[1:], yy[1:], color=P.SLOTS[0], lw=1.0, ls="--", drawstyle="steps-post", marker="x", ms=5,
                    label="log(ρ·b_t), median, ρ=0.1")
            trig = [s["t"] for s in r["trace"] if s["log_p"] is not None]
            ax.set_xlabel("demonstration index t")
            ax.set_ylabel("log probability")
            if p == "original":
                ax.set_xlim(1.5, 3.5)
            ax.set_title(f"{'Exp. 1' if d == 'cart' else 'Exp. 2'} – {'original D1–D3' if p == 'original' else 'pool seed 0'}")
            ax.set_xticks(ts)
            if len(ts) > 10:
                ax.set_xticks(ts[::2])
    axes[0][0].legend(fontsize=6.5)
    p_ = os.path.join(fig_dir, "threshold_traces.png")
    P.save(fig, p_)
    paths.append(p_)
    # ---- log p_t versus demonstration length (running-minimum weakness)
    fig, axes = P.plt.subplots(1, len(domains), figsize=(4.6 * len(domains), 3.3), squeeze=False)
    for ax, d in zip(axes[0], domains):
        xs, ys = [], []
        for r in rows:
            if r["domain"] == d and r["protocol"] == "pool" and r["method"] == "fixed":
                for s in r["trace"]:
                    if s["log_p"] is not None:
                        xs.append(s["demo_len"]); ys.append(s["log_p"])
        ax.scatter(xs, ys, s=10, color=P.SLOTS[0], alpha=0.5, edgecolor="none")
        ax.set_xlabel("demonstration length (task cost)")
        ax.set_ylabel("log p_t")
        ax.set_title({"cart": "Exp. 1: cart-use", "aisle": "Exp. 2: aisle-use"}[d])
    p_ = os.path.join(fig_dir, "threshold_logp_vs_length.png")
    P.save(fig, p_)
    paths.append(p_)
    return paths


def main():
    cfg = parse_args("Refinement-threshold sensitivity")
    od = out_dir(cfg, "threshold_sensitivity")
    write_json(os.path.join(od, "metadata.json"), run_metadata("threshold_sensitivity", cfg))
    t0 = time.time()
    jobs = build_jobs(cfg)
    print(f"{len(jobs)} runs")
    rows = pmap(_job, jobs, cfg["workers"])
    write_jsonl(os.path.join(od, "runs.jsonl"), rows)
    write_csv(os.path.join(od, "runs.csv"), [{k: v for k, v in r.items() if k != "trace"} for r in rows])
    trace_rows = [{"domain": r["domain"], "protocol": r["protocol"], "order_or_seed": r["order_or_seed"],
                   "method": r["method"], "rho": r["rho"], "baseline": r["baseline"], **s}
                  for r in rows for s in r["trace"]]
    write_csv(os.path.join(od, "traces.csv"), trace_rows)
    summary = summarise(rows, cfg)
    write_csv(os.path.join(od, "summary.csv"), summary)
    diag = running_min_diagnostics(rows)
    write_json(os.path.join(od, "running_min_diagnostics.json"), diag)
    figs = plots(summary, rows, cfg, FIGURES)
    write_json(os.path.join(od, "outputs.json"), {"figures": figs, "runtime_s": time.time() - t0})
    for s in summary:
        if s["method"] == "full":
            print(f"{s['domain']:5s} {s['protocol']:8s} rho={s['rho']:<6} {s['baseline']:6s} acc={s['accuracy_mean']:.3f} "
                  f"f1={s['violation_f1_mean']:.3f} refrate={s['refinement_rate']:.2f} "
                  f"P_int={s.get('P_intended_mean', float('nan')):.3f}")
    print(diag)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
