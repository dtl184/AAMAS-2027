"""NEW EXPERIMENT 2: robustness to noisy demonstrations.

For each domain, noise type (behavioural / violation), noise rate q and seed, a training sequence of
`noise_demos` demonstrations is generated from the task pool; round(q * n) randomly chosen positions are
corrupted.  Clean demonstrations (tasks, tie-breaking, corrupted positions) are shared across rates and
noise types for the same seed (paired design; corrupted sets are nested in q).  The held-out set is the
fixed clean set of the reconstruction.  Methods: full, fixed, oracle, MLCI.

  behavioural noise: norm-compliant but inefficient (random out-and-back detours, redundant
                     leave/pick-up of the cart); the hidden-norm label is unchanged (verified).
  violation noise:   the demonstration is replaced by a valid trajectory that violates the hidden norm
                     (cart: no return / wrong location / return-then-reacquire; aisle: cart into aisle).

python -m experiments.noise_robustness [--quick] [--set noise_seeds=20 noise_demos=20]
"""
from __future__ import annotations

import os
import time

import numpy as np

from analysis import plotting as P
from environment.domains import get_spec
from experiments.common import (FIGURES, bootstrap_ci, flat, make_pool_dataset, out_dir, parse_args, pmap,
                                run_metadata, run_method, seed_for, write_csv, write_json, write_jsonl)

NOISE_TYPES = ["behavioural", "violation"]


def _job(job):
    key, ntype, rate, seed_idx, method, cfg = job
    spec = get_spec(key)
    seed = seed_for(cfg, "noise", key, seed_idx)
    demos = make_pool_dataset(spec, cfg["noise_demos"], seed, None if rate == 0 else ntype, rate)
    r = run_method(spec, method, demos, cfg)
    corrupted = [i + 1 for i, t in enumerate(demos) if t.meta.get("corrupted")]
    return {"domain": key, "noise_type": ntype, "rate": rate, "seed": seed_idx, "dataset_seed": seed,
            "method": method, **flat(r), "n_demos": len(demos), "n_corrupted": len(corrupted),
            "corrupted_positions": corrupted,
            "violation_types": [t.meta.get("violation_type") for t in demos if t.meta.get("violation_type")],
            "refinement_points": r.get("refinement_points"),
            "n_true_violating_demos": int(sum(spec.violates(t) for t in demos)),
            "dataset": [{"name": t.name, "items": list(t.task.items), "kind": t.meta.get("kind"),
                         "corrupted": bool(t.meta.get("corrupted")), "length": len(t.actions),
                         "actions": [spec.domain.action_name(a) for a in t.actions]} for t in demos]
            if method == "full" else None}


def build_jobs(cfg):
    jobs = []
    for key in cfg["domains"]:
        for s in range(cfg["noise_seeds"]):
            for m in cfg["noise_methods"]:
                if 0.0 in cfg["noise_rates"]:
                    jobs.append((key, "none", 0.0, s, m, cfg))
                for nt in NOISE_TYPES:
                    for q in cfg["noise_rates"]:
                        if q > 0:
                            jobs.append((key, nt, q, s, m, cfg))
    return jobs


def summarise(rows, cfg):
    out = []
    for key in cfg["domains"]:
        for nt in NOISE_TYPES:
            for q in cfg["noise_rates"]:
                for m in cfg["noise_methods"]:
                    rs = [r for r in rows if r["domain"] == key and r["method"] == m and r["rate"] == q
                          and (r["noise_type"] == nt or (q == 0 and r["noise_type"] == "none"))]
                    if not rs:
                        continue
                    row = {"domain": key, "noise_type": nt, "rate": q, "method": m, "n_seeds": len(rs),
                           "mean_n_corrupted": float(np.mean([r["n_corrupted"] for r in rs]))}
                    for metric in ("accuracy", "violation_f1", "unseen_accuracy", "P_intended", "n_refinements",
                                   "n_hypotheses", "runtime_s"):
                        if metric == "P_intended" and m == "mlci":
                            continue
                        vals = ([r.get(metric) or 0.0 for r in rs] if metric == "P_intended"
                                else [r[metric] for r in rs if r.get(metric) is not None])
                        if not vals:
                            continue
                        mu, lo, hi = bootstrap_ci(vals, cfg["bootstrap"], seed=1)
                        row.update({f"{metric}_mean": mu, f"{metric}_sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                                    f"{metric}_se": float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0,
                                    f"{metric}_ci_lo": lo, f"{metric}_ci_hi": hi})
                    if m != "mlci":
                        row["map_intended_rate"] = float(np.mean([bool(r.get("map_is_intended")) for r in rs]))
                        row["map_equivalent_rate"] = float(np.mean([bool(r.get("map_equivalent_on_heldout")) for r in rs]))
                        row["refinement_rate"] = float(np.mean([(r.get("n_refinements") or 0) > 0 for r in rs]))
                        maps = {}
                        for r in rs:
                            maps[r["map_id"]] = maps.get(r["map_id"], 0) + 1
                        row["map_counts"] = maps
                    else:
                        row["mean_n_constraints"] = float(np.mean([r.get("n_constraints") or 0 for r in rs]))
                    out.append(row)
    return out


def plots(summary, cfg):
    P.setup()
    paths = []
    rates = cfg["noise_rates"]
    for nt in NOISE_TYPES:
        metrics = [("accuracy", "Held-out accuracy"), ("violation_f1", "Violation F1")]
        if "aisle" in cfg["domains"]:
            metrics.append(("unseen_accuracy", "Unseen-aisle accuracy (Exp. 2)"))
        fig, axes = P.plt.subplots(len(metrics), len(cfg["domains"]), figsize=(4.6 * len(cfg["domains"]), 3.0 * len(metrics)),
                                   squeeze=False)
        for i, (metric, ylab) in enumerate(metrics):
            for j, key in enumerate(cfg["domains"]):
                ax = axes[i][j]
                if metric == "unseen_accuracy" and key == "cart":
                    ax.axis("off")
                    continue
                for m in cfg["noise_methods"]:
                    rs = [next((s for s in summary if s["domain"] == key and s["noise_type"] == nt and s["rate"] == q
                                and s["method"] == m), None) for q in rates]
                    if any(r is None or f"{metric}_mean" not in r for r in rs):
                        continue
                    x = np.array(rates) * 100
                    P.mean_ci_line(ax, x, [r[f"{metric}_mean"] for r in rs], [r[f"{metric}_ci_lo"] for r in rs],
                                   [r[f"{metric}_ci_hi"] for r in rs], P.METHOD_STYLE[m])
                ax.set_ylim(-0.03, 1.03)
                ax.set_xlabel(f"{'behavioural' if nt == 'behavioural' else 'norm-violation'} noise (% of demonstrations)")
                ax.set_ylabel(ylab)
                ax.set_title({"cart": "Exp. 1: cart-use", "aisle": "Exp. 2: aisle-use"}[key])
        axes[0][-1].legend(fontsize=7, loc="lower left")
        p_ = os.path.join(FIGURES, f"noise_{nt}.png")
        P.save(fig, p_)
        paths.append(p_)
    return paths


def main():
    cfg = parse_args("Robustness to noisy demonstrations")
    od = out_dir(cfg, "noise_robustness")
    write_json(os.path.join(od, "metadata.json"), run_metadata("noise_robustness", cfg))
    t0 = time.time()
    jobs = build_jobs(cfg)
    print(f"{len(jobs)} runs")
    rows = pmap(_job, jobs, cfg["workers"])
    write_jsonl(os.path.join(od, "runs.jsonl"), rows)
    write_csv(os.path.join(od, "runs.csv"), [{k: v for k, v in r.items() if k != "dataset"} for r in rows])
    summary = summarise(rows, cfg)
    write_csv(os.path.join(od, "summary.csv"), summary)
    figs = plots(summary, cfg)
    write_json(os.path.join(od, "outputs.json"), {"figures": figs, "runtime_s": time.time() - t0})
    for s in summary:
        print(f"{s['domain']:5s} {s['noise_type']:11s} q={s['rate']:.2f} {s['method']:6s} "
              f"acc={s['accuracy_mean']:.3f}±{s['accuracy_se']:.3f} f1={s['violation_f1_mean']:.3f} "
              f"unseen={s.get('unseen_accuracy_mean', float('nan')):.3f} "
              f"Pint={s.get('P_intended_mean', float('nan')):.3f} maprate={s.get('map_intended_rate', float('nan'))}")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
