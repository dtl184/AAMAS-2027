"""Experiment 1 on NEW demonstration sequences: three adaptive refinement policies, frozen GPT-5.5 proposals.

Reuses experiments/noise_selected_trigger.py unchanged (its _job: FrozenProposalProvider replaying the cached GPT-5.5
proposal sets of results/gpt55/refinement_threshold_extended, NormLearner, make_pool_dataset, evaluation).  The ONLY
differences from Experiment 2 are the seed indices (100-119 instead of 0-19) and the output directory.  No OpenAI
calls: the API key is removed from the process by the imported module and no client is constructed.

Policies: adaptive_raw_rho0.1 (paper's raw predictive trigger, rho = 0.1), adaptive_task_q0.1 (task-cost-adjusted,
quantile q = 0.10), adaptive_rel_q0.1 (norm-relative, q = 0.10).  Fixed and oracle abstraction are run as references.
Settings unchanged: beta = 2, lambda = gamma = 0.2, W = {0, 0.5, ..., 30}, 20 demonstrations per run, both domains,
conditions q = 0 (primary) plus behavioural / violation noise at q in {0.05, 0.10, 0.20, 0.30}.
Every method receives the identical demonstrations for a given (domain, seed, condition).

    cd /home/train/norm_discovery && /home/train/anaconda3/bin/python -m experiments.exp1_policy_heldout
"""
from __future__ import annotations

import collections
import json
import os
import sys
import time

import experiments.noise_selected_trigger as NST       # pops OPENAI_API_KEY at import

import numpy as np

from analysis import plotting as P
from environment.domains import get_spec
from experiments.common import (DEFAULTS, RESULTS, ROOT, make_pool_dataset, parse_args, pmap, run_metadata, seed_for,
                                write_csv, write_json, write_jsonl)

OUT = os.path.join(RESULTS, "exp1_policy_heldout")
FIG = os.path.join(ROOT, "figures", "exp1_policy_heldout")
SEEDS = list(range(100, 120))
OLD_SEEDS = list(range(0, 20))
POLICIES = ["adaptive_raw_rho0.1", "adaptive_task_q0.1", "adaptive_rel_q0.1"]
REFERENCES = ["fixed", "oracle"]
LABEL = {"adaptive_raw_rho0.1": "Raw predictive (ρ=0.1)", "adaptive_task_q0.1": "Task-cost-adjusted (q=0.10)",
         "adaptive_rel_q0.1": "Norm-relative (q=0.10)", "fixed": "Fixed abstraction (ref.)",
         "oracle": "Oracle abstraction (ref.)"}


def conditions(cfg):
    return [(None, 0.0)] + [(nt, q) for nt in NST.NOISE_TYPES for q in cfg["noise_rates"] if q > 0]


def dataset_signature(spec, cfg, seed, nt, q):
    demos = make_pool_dataset(spec, cfg["noise_demos"], seed_for(cfg, "noise", spec.key, seed), None if q == 0 else nt, q)
    return [(t.task.items, t.task.start, tuple(t.actions)) for t in demos]


def verify_new_data(cfg):
    """Training sequences for seeds 100-119 must differ from Experiment 2's (seeds 0-19)."""
    rep = {}
    for key in ("cart", "aisle"):
        spec = get_spec(key)
        for nt, q in conditions(cfg):
            old = [dataset_signature(spec, cfg, s, nt, q) for s in OLD_SEEDS]
            new = [dataset_signature(spec, cfg, s, nt, q) for s in SEEDS]
            old_seqs = {tuple(x) for x in old}
            old_demos = collections.Counter(d for x in old for d in x)
            new_demos = [d for x in new for d in x]
            rep[f"{key}/{nt or 'none'}/{q}"] = {
                "identical_sequences_with_exp2": sum(tuple(x) in old_seqs for x in new),
                "new_demos": len(new_demos),
                "new_demos_also_occurring_in_exp2": sum(d in old_demos for d in new_demos),
                "distinct_dataset_seeds_overlap": len({seed_for(cfg, "noise", key, s) for s in SEEDS}
                                                      & {seed_for(cfg, "noise", key, s) for s in OLD_SEEDS})}
    return rep


def boot(v, n=4000, seed=0):
    v = np.asarray([x for x in v if x is not None], float)
    if not len(v):
        return (None, None, None)
    bs = np.random.default_rng(seed).choice(v, size=(n, len(v)), replace=True).mean(1)
    return float(v.mean()), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))


def fmt(t, nd=2):
    return "–" if t[0] is None else f"{t[0]:.{nd}f} [{t[1]:.{nd}f}, {t[2]:.{nd}f}]"


def main():
    DEFAULTS.update({"noise_seeds": 20})
    cfg = parse_args("Experiment 1 on held-out seeds 100-119")
    cfg["provider"] = "frozen_gpt55"
    if os.environ.get("OPENAI_API_KEY"):
        sys.exit("ABORT: API key present in process")
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    t0 = time.time()
    cache_before = len(os.listdir(os.path.join(RESULTS, "llm_cache")))
    reps = {k: sorted(NST._frozen()[k]) for k in ("cart", "aisle")}
    assert all(len(v) == 10 for v in reps.values())
    data_check = verify_new_data(cfg)
    assert all(v["identical_sequences_with_exp2"] == 0 and v["distinct_dataset_seeds_overlap"] == 0
               for v in data_check.values()), "new seeds reproduce Experiment 2 sequences"
    jobs = [(cfg, k, s, nt, q, m) for k in ("cart", "aisle") for s in SEEDS for nt, q in conditions(cfg)
            for m in POLICIES + REFERENCES]
    plan = {"seeds": SEEDS, "conditions": len(conditions(cfg)), "methods": POLICIES + REFERENCES,
            "inference_runs": len(jobs), "openai_calls_required": 0,
            "frozen_replicates": {k: len(v) for k, v in reps.items()}, "replicate_for_seed": "seed mod 10"}
    print(json.dumps(plan, indent=1))
    meta = run_metadata("exp1_policy_heldout", cfg)
    meta.update({"plan": plan, "evaluates": NST.__doc__.split("What this experiment evaluates:")[1].split("\n\n")[0].strip(),
                 "frozen_proposal_source": NST.FROZEN_SRC})
    write_json(os.path.join(OUT, "metadata.json"), meta)
    write_json(os.path.join(OUT, "data_independence_check.json"), data_check)
    rows = pmap(NST._job, jobs, cfg["workers"])
    # paired-design check: every method saw the identical corrupted positions for each (domain, seed, condition)
    grp = collections.defaultdict(set)
    for r in rows:
        grp[(r["domain"], r["seed"], r["noise_type"], r["rate"])].add(tuple(r["corrupted_positions"]))
    paired_ok = all(len(v) == 1 for v in grp.values())
    write_jsonl(os.path.join(OUT, "runs.jsonl"), rows)
    write_csv(os.path.join(OUT, "runs.csv"), [{k: v for k, v in r.items() if k != "trigger_events"} for r in rows])
    write_csv(os.path.join(OUT, "trigger_events.csv"),
              [{"domain": r["domain"], "seed": r["seed"], "noise_type": r["noise_type"], "rate": r["rate"],
                "method": r["method"], **e} for r in rows for e in r["trigger_events"]])
    ex = NST.expand(rows)
    summ, paired = [], []
    for key in ("cart", "aisle"):
        for nt in NST.NOISE_TYPES:
            for q in cfg["noise_rates"]:
                sel = lambda m: sorted([r for r in ex if r["domain"] == key and r["noise_type"] == nt and r["rate"] == q
                                        and r["method"] == m], key=lambda r: r["seed"])
                for m in POLICIES + REFERENCES:
                    rs = sel(m)
                    row = {"domain": key, "noise_type": nt, "rate": q, "method": m, "n_seeds": len(rs)}
                    for mt in ("accuracy", "violation_f1", "unseen_accuracy", "P_intended", "n_refinements",
                               "first_refinement_t", "w_map"):
                        mu, lo, hi = boot([r.get(mt) for r in rs])
                        row.update({f"{mt}_mean": mu, f"{mt}_lo": lo, f"{mt}_hi": hi})
                    row["refinement_rate"] = float(np.mean([(r["n_refinements"] or 0) > 0 for r in rs]))
                    row["map_intended_rate"] = float(np.mean([bool(r["map_is_intended"]) for r in rs]))
                    summ.append(row)
                for a, b in (("adaptive_task_q0.1", "adaptive_raw_rho0.1"), ("adaptive_rel_q0.1", "adaptive_raw_rho0.1"),
                             ("adaptive_rel_q0.1", "adaptive_task_q0.1")):
                    A, B = {r["seed"]: r for r in sel(a)}, {r["seed"]: r for r in sel(b)}
                    for mt in ("accuracy", "violation_f1", "unseen_accuracy", "n_refinements", "P_intended"):
                        d = [A[s][mt] - B[s][mt] for s in A if s in B and A[s][mt] is not None and B[s][mt] is not None]
                        if d:
                            paired.append({"domain": key, "noise_type": nt, "rate": q, "comparison": f"{a} - {b}",
                                           "metric": mt, "diff": fmt(boot(d)), "diff_mean": float(np.mean(d)),
                                           "n_pairs": len(d)})
    write_csv(os.path.join(OUT, "summary.csv"), summ)
    write_csv(os.path.join(OUT, "paired_differences.csv"), paired)
    checks = {"data_independent_of_exp2": True, "paired_design_identical_demonstrations": paired_ok,
              "llm_cache_files_before": cache_before, "llm_cache_files_after": len(os.listdir(os.path.join(RESULTS, "llm_cache"))),
              "api_key_in_process": bool(os.environ.get("OPENAI_API_KEY"))}
    write_json(os.path.join(OUT, "validation.json"), checks)
    figures(summ, cfg)
    print(json.dumps(checks))
    for s in summ:
        if s["rate"] in (0.0, 0.3):
            print(f"{s['domain']:5s} {s['noise_type']:11s} q={s['rate']:.2f} {s['method']:20s} "
                  f"acc={fmt((s['accuracy_mean'], s['accuracy_lo'], s['accuracy_hi']))} F1={s['violation_f1_mean']:.2f} "
                  f"unseen={s['unseen_accuracy_mean']} nref={s['n_refinements_mean']:.2f} refrate={s['refinement_rate']:.2f} "
                  f"first_t={s['first_refinement_t_mean']} P={s['P_intended_mean']:.2f}")
    for p in paired:
        if p["rate"] in (0.0, 0.3) and p["metric"] in ("accuracy", "n_refinements", "unseen_accuracy"):
            print(p["domain"], p["noise_type"], p["rate"], p["comparison"], p["metric"], p["diff"])
    print(f"done in {time.time() - t0:.0f}s")


def figures(summ, cfg):
    P.setup()
    col = {"adaptive_raw_rho0.1": P.SLOTS[6], "adaptive_task_q0.1": P.SLOTS[0], "adaptive_rel_q0.1": P.SLOTS[4],
           "fixed": P.SLOTS[1], "oracle": P.SLOTS[2]}
    mk = {"adaptive_raw_rho0.1": "v", "adaptive_task_q0.1": "o", "adaptive_rel_q0.1": "D", "fixed": "s", "oracle": "^"}
    for metric, ylab, fname in (("accuracy", "held-out accuracy", "accuracy"), ("n_refinements", "refinements per run", "refinements")):
        fig, axes = P.plt.subplots(2, 2, figsize=(10, 6.4), squeeze=False)
        for i, nt in enumerate(NST.NOISE_TYPES):
            for j, key in enumerate(("cart", "aisle")):
                ax = axes[i][j]
                for m in POLICIES + (REFERENCES if metric == "accuracy" else []):
                    ss = sorted([s for s in summ if s["domain"] == key and s["noise_type"] == nt and s["method"] == m],
                                key=lambda s: s["rate"])
                    style = {"color": col[m], "marker": mk[m], "label": LABEL[m]}
                    P.mean_ci_line(ax, [s["rate"] * 100 for s in ss], [s[f"{metric}_mean"] for s in ss],
                                   [s[f"{metric}_lo"] for s in ss], [s[f"{metric}_hi"] for s in ss], style)
                if metric == "accuracy":
                    ax.set_ylim(-0.03, 1.03)
                ax.set_xlabel(f"{nt} noise (% of 20 demos)"); ax.set_ylabel(ylab)
                ax.set_title({"cart": "Exp. 1: cart-use", "aisle": "Exp. 2: aisle-use"}[key] + " – seeds 100–119")
        axes[0][1].legend(fontsize=6.5, loc="lower right")
        P.save(fig, os.path.join(FIG, f"policies_{fname}.png"))


if __name__ == "__main__":
    main()
