"""Robustness to imperfect demonstrations with the selected refinement trigger and FROZEN GPT-5.5 proposals.

NO OpenAI calls are made.  OPENAI_API_KEY is removed from this process's environment and no OpenAI client is ever
constructed: hypotheses come from FrozenProposalProvider, which replays proposal sets that GPT-5.5 already generated
(results/gpt55/refinement_threshold_extended, rho = 20000: every run there has its initial (alpha_0) set and its
refined set from the original demonstrations D1-D3).  Replicate k (k = 0..9) = GPT-5.5 run k of that domain;
noise seed s uses replicate s mod 10.

What this experiment evaluates: trigger robustness + Bayesian-inference robustness under noise, given real GPT-5.5
proposal sets.  It does NOT evaluate whether GPT-5.5 would propose different hypotheses when shown noisy
demonstrations (the refined sets were proposed from the clean original demonstrations).

Methods (identical datasets for all methods; same datasets/seeds as results/deterministic_proposals/noise_robustness):
  adaptive_task_q0.1   PRIMARY: task-cost-adjusted trigger  log p_t + beta C(tau_t) < Q_0.10(accepted scores)
  adaptive_task_q0.05, adaptive_task_q0.2   threshold sensitivity of the primary trigger
  adaptive_rel_q0.1    secondary: norm-relative trigger (log p_t - log P_task(tau_t))
  adaptive_raw_rho0.1  the paper's original raw predictive trigger p_t < 0.1 b_t (for comparison only)
  fixed                initial GPT-5.5 hypotheses only, never refined
  oracle               initial + refined GPT-5.5 hypotheses from t = 1
  mlci                 reused from results/deterministic_proposals/noise_robustness (identical datasets & settings;
                       spot-checked by recomputation)

    cd /home/train/norm_discovery && /home/train/anaconda3/bin/python -m experiments.noise_selected_trigger
"""
from __future__ import annotations

import collections
import json
import os
import sys
import time

os.environ.pop("OPENAI_API_KEY", None)        # hard guarantee: no paid API call can be made from this process

import numpy as np

from analysis import plotting as P
from environment.domains import get_spec
from experiments.common import (DEFAULTS, RESULTS, ROOT, make_pool_dataset, make_zc, parse_args, pmap, run_metadata,
                                run_method, seed_for, write_csv, write_json, write_jsonl)
from proposals.base import ProposalProvider

OUT = os.path.join(RESULTS, "noise_selected_trigger")
FIG = os.path.join(ROOT, "figures", "noise_selected_trigger")
FROZEN_SRC = os.path.join(RESULTS, "gpt55", "refinement_threshold_extended", "runs.jsonl")
N_REPLICATES = 10
NOISE_TYPES = ("behavioural", "violation")
METHODS = {
    "adaptive_task_q0.1": dict(trigger_family="task", trigger_rule="quantile", trigger_param=0.10),
    "adaptive_task_q0.05": dict(trigger_family="task", trigger_rule="quantile", trigger_param=0.05),
    "adaptive_task_q0.2": dict(trigger_family="task", trigger_rule="quantile", trigger_param=0.20),
    "adaptive_rel_q0.1": dict(trigger_family="rel", trigger_rule="quantile", trigger_param=0.10),
    "adaptive_raw_rho0.1": dict(trigger_family="raw", trigger_rule="ratio", trigger_param=None, rho=0.1),
    "fixed": dict(refinement=False),
    "oracle": dict(refinement=False, oracle=True),
}
PRIMARY = "adaptive_task_q0.1"
STYLE = {"adaptive_task_q0.1": {"color": P.SLOTS[0], "marker": "o", "label": "Adaptive refinement (task-adjusted trigger, q=0.10)"},
         "fixed": dict(P.METHOD_STYLE["fixed"], label="Fixed abstraction"),
         "oracle": dict(P.METHOD_STYLE["oracle"], label="Oracle abstraction"),
         "mlci": dict(P.METHOD_STYLE["mlci"], label="MLCI"),
         "adaptive_raw_rho0.1": {"color": P.SLOTS[6], "marker": "v", "label": "Old raw trigger (ρ=0.1)"},
         "adaptive_rel_q0.1": {"color": P.SLOTS[4], "marker": "D", "label": "Norm-relative trigger (q=0.10)"}}


class FrozenProposalProvider(ProposalProvider):
    """Replays a cached GPT-5.5 proposal replicate; never calls any model."""
    name = "frozen_gpt55"
    max_level = 1

    def __init__(self, initial, refined, replicate):
        self.initial, self.refined, self.replicate = initial, refined, replicate
        self.n_refine_calls = 0
        self.transcript = []

    def propose_initial(self, spec, demos):
        self.transcript.append({"kind": "initial", "replicate": self.replicate, "n": len(self.initial)})
        return [dict(h) for h in self.initial]

    def refine(self, spec, demos, current, level):
        self.n_refine_calls += 1
        out = [dict(h) for h in self.refined] if self.n_refine_calls == 1 else []   # no new proposals after the first
        self.transcript.append({"kind": "refinement", "replicate": self.replicate, "call": self.n_refine_calls,
                                "n_returned": len(out), "n_demos_seen": len(demos)})
        return out


def load_frozen():
    by = collections.defaultdict(dict)
    with open(FROZEN_SRC) as f:
        for l in f:
            r = json.loads(l)
            if r["rho"] != 20000 or r.get("error") or r["n_refinements"] != 1:
                continue
            ini = [h["json"] for h in r["all_hypotheses"] if h["level"] == 0]
            ref = [h["json"] for h in r["all_hypotheses"] if h["level"] == 1]
            if ini and ref:
                by[r["domain"]][r["run_index"]] = {"initial": ini, "refined": ref,
                                                   "models": r.get("returned_models")}
    return by


FROZEN = None


def _frozen():
    global FROZEN
    if FROZEN is None:
        FROZEN = load_frozen()
    return FROZEN


def _job(job):
    from inference.refinement import LearnerConfig, NormLearner
    cfg, key, seed, ntype, q, method = job
    spec = get_spec(key)
    demos = make_pool_dataset(spec, cfg["noise_demos"], seed_for(cfg, "noise", key, seed), None if q == 0 else ntype, q)
    rep = seed % N_REPLICATES
    fr = _frozen()[key][rep]
    prov = FrozenProposalProvider(fr["initial"], fr["refined"], rep)
    kw = dict(METHODS[method])
    lc = LearnerConfig(**kw)
    L = NormLearner(spec, prov, lc, make_zc(spec, cfg))
    t0 = time.time()
    res = L.run(demos, record_prefix_eval=False)
    f = res["final"]
    w_map = next((row["E_w"] for row in f["posterior_table"] if row["id"] == f["map_id"]), None)
    meta = [{"t": i + 1, "corrupted": bool(t.meta.get("corrupted")), "kind": t.meta.get("kind"),
             "violation_type": t.meta.get("violation_type"), "violates_hidden_norm": bool(spec.violates(t)),
             "length": len(t.actions), "C_task": t.cost} for i, t in enumerate(demos)]
    events = []
    for s in res["trace"][1:]:
        m = meta[s["t"] - 1]
        events.append({**m, "trigger_family": s.get("trigger_family"), "trigger_rule": s.get("trigger_rule"),
                       "trigger_score": s.get("trigger_score"), "trigger_baseline": s.get("trigger_baseline"),
                       "trigger_threshold": s.get("trigger_threshold"), "triggered": s["triggered"],
                       "n_refinements_so_far": s.get("n_refinements_so_far"), "P_intended": s["P_intended"] or 0.0,
                       **{f"score_{k}": v for k, v in (s.get("scores") or {}).items()}})
    return {"domain": key, "seed": seed, "frozen_replicate": rep, "noise_type": ntype if q > 0 else "none", "rate": q,
            "method": method, "accuracy": f["accuracy"], "violation_f1": f["violation_f1"],
            "unseen_accuracy": f["unseen_accuracy"], "map_id": f["map_id"], "map_description": f["map_description"],
            "map_is_intended": f["map_is_intended"], "P_intended": f["P_intended"] or 0.0, "w_map": w_map,
            "n_refinements": f["n_refinements"], "refinement_points": f["refinement_points"],
            "first_refinement_t": f["refinement_points"][0] if f["refinement_points"] else None,
            "n_hypotheses_end": f["n_hypotheses"], "n_corrupted": sum(m["corrupted"] for m in meta),
            "corrupted_positions": [m["t"] for m in meta if m["corrupted"]], "trigger_events": events,
            "runtime_s": time.time() - t0, "proposals": "frozen GPT-5.5 replicate (no API call)"}


def _mlci_reuse(cfg):
    """MLCI rows from the deterministic noise experiment (identical datasets and settings)."""
    rows = []
    with open(os.path.join(RESULTS, "deterministic_proposals", "noise_robustness", "runs.jsonl")) as f:
        for l in f:
            r = json.loads(l)
            if r["method"] != "mlci":
                continue
            rows.append({"domain": r["domain"], "seed": r["seed"], "frozen_replicate": None, "noise_type": r["noise_type"],
                         "rate": r["rate"], "method": "mlci", "accuracy": r["accuracy"], "violation_f1": r["violation_f1"],
                         "unseen_accuracy": r.get("unseen_accuracy"), "map_id": None, "map_is_intended": None,
                         "P_intended": None, "w_map": None, "n_refinements": None, "n_hypotheses_end": r.get("n_constraints"),
                         "n_corrupted": r["n_corrupted"], "corrupted_positions": r["corrupted_positions"],
                         "proposals": "none (MLCI); reused from results/deterministic_proposals/noise_robustness"})
    return rows


def _mlci_check(job):
    cfg, key, seed, ntype, q = job
    spec = get_spec(key)
    demos = make_pool_dataset(spec, cfg["noise_demos"], seed_for(cfg, "noise", key, seed), None if q == 0 else ntype, q)
    r = run_method(spec, "mlci", demos, cfg)
    return {"domain": key, "seed": seed, "noise_type": ntype if q > 0 else "none", "rate": q, "accuracy": r["accuracy"],
            "violation_f1": r["violation_f1"], "constraints": r["constraints"]}


# ------------------------------------------------------------------------------------------------- statistics
def boot(vals, n=4000, seed=0):
    v = np.asarray([x for x in vals if x is not None], float)
    if len(v) == 0:
        return (None, None, None)
    rng = np.random.default_rng(seed)
    bs = rng.choice(v, size=(n, len(v)), replace=True).mean(1)
    return float(v.mean()), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))


def fmt(t, nd=2):
    return "–" if t[0] is None else f"{t[0]:.{nd}f} [{t[1]:.{nd}f}, {t[2]:.{nd}f}]"


def expand(rows):
    """q = 0 belongs to both noise-type curves."""
    out = []
    for r in rows:
        if r["noise_type"] == "none":
            out += [dict(r, noise_type=nt) for nt in NOISE_TYPES]
        else:
            out.append(r)
    return out


def main():
    DEFAULTS.update({"noise_seeds": 20})
    cfg = parse_args("Selected-trigger noise robustness (frozen GPT-5.5 proposals, no API calls)")
    cfg["provider"] = "frozen_gpt55"
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    fr = _frozen()
    reps = {k: sorted(fr[k]) for k in ("cart", "aisle")}
    conds = [(None, 0.0)] + [(nt, q) for nt in NOISE_TYPES for q in cfg["noise_rates"] if q > 0]
    jobs = [(cfg, k, s, nt, q, m) for k in ("cart", "aisle") for s in range(cfg["noise_seeds"]) for nt, q in conds
            for m in METHODS if s % N_REPLICATES in reps[k]]
    plan = {"domains": 2, "noise_conditions_per_domain": len(conds), "seeds": cfg["noise_seeds"],
            "learner_methods": list(METHODS), "planned_conditions": 2 * len(conds) * (len(METHODS) + 1),
            "inference_runs": len(jobs), "mlci_runs_reused": 2 * len(conds) * cfg["noise_seeds"],
            "frozen_gpt55_replicates_available": {k: len(v) for k, v in reps.items()},
            "openai_calls_required": 0}
    print(json.dumps(plan, indent=1))
    if plan["openai_calls_required"] != 0 or os.environ.get("OPENAI_API_KEY"):
        sys.exit("ABORT: this experiment must not make OpenAI calls")
    meta = run_metadata("noise_selected_trigger", cfg)
    meta["evaluates"] = ("trigger robustness + Bayesian inference robustness under noise, using frozen real GPT-5.5 "
                         "proposal sets; it does NOT evaluate whether GPT-5.5 would propose differently from noisy "
                         "demonstrations")
    meta["frozen_proposal_source"] = FROZEN_SRC
    meta["frozen_replicate_models"] = {k: sorted({m for r in fr[k].values() for m in (r["models"] or [])}) for k in fr}
    meta["selected_trigger"] = ("No trigger was finalised by the (incomplete) trigger-comparison experiment; per the "
                                "protocol the task-cost-adjusted trigger is primary.  Threshold q = 0.10 fixed a priori "
                                "(middle of that experiment's pre-registered quantile grid); q = 0.05 / 0.20 reported as "
                                "sensitivity; norm-relative (q = 0.10) as secondary; the raw trigger for comparison.")
    meta["plan"] = plan
    write_json(os.path.join(OUT, "metadata.json"), meta)
    t0 = time.time()
    rows = pmap(_job, jobs, cfg["workers"])
    # MLCI: reuse + spot-check by recomputation
    mlci = _mlci_reuse(cfg)
    checks = pmap(_mlci_check, [(cfg, "cart", 3, "violation", 0.2), (cfg, "aisle", 5, "behavioural", 0.1),
                                (cfg, "aisle", 0, None, 0.0)], cfg["workers"])
    check_ok = all(any(m["domain"] == c["domain"] and m["seed"] == c["seed"] and m["rate"] == c["rate"]
                       and m["noise_type"] == c["noise_type"] and m["accuracy"] == c["accuracy"]
                       and m["violation_f1"] == c["violation_f1"] for m in mlci) for c in checks)
    write_json(os.path.join(OUT, "mlci_reuse_check.json"), {"recomputed": checks, "identical_to_reused": check_ok})
    if not check_ok:
        print("MLCI reuse check failed -> recomputing all MLCI runs")
        mlci = [dict(r, method="mlci", proposals="none (MLCI); recomputed") for r in
                pmap(_mlci_check, [(cfg, k, s, nt, q) for k in ("cart", "aisle") for s in range(cfg["noise_seeds"])
                                   for nt, q in conds], cfg["workers"])]
    allrows = rows + mlci
    write_jsonl(os.path.join(OUT, "runs.jsonl"), allrows)
    write_csv(os.path.join(OUT, "runs.csv"), [{k: v for k, v in r.items() if k != "trigger_events"} for r in allrows])
    ev = [{"domain": r["domain"], "seed": r["seed"], "noise_type": r["noise_type"], "rate": r["rate"],
           "method": r["method"], **e} for r in rows if r["method"].startswith("adaptive") or r["method"] == "fixed"
          for e in r["trigger_events"]]
    write_csv(os.path.join(OUT, "trigger_events.csv"), ev)
    analyse(allrows, ev, cfg)
    print(f"done in {time.time() - t0:.0f}s; MLCI reuse check identical: {check_ok}")


# ------------------------------------------------------------------------------------------------- analysis
def analyse(rows, ev, cfg):
    ex = expand(rows)
    methods = list(METHODS) + ["mlci"]
    summ = []
    for key in ("cart", "aisle"):
        for nt in NOISE_TYPES:
            for q in cfg["noise_rates"]:
                for m in methods:
                    rs = [r for r in ex if r["domain"] == key and r["noise_type"] == nt and r["rate"] == q and r["method"] == m]
                    if not rs:
                        continue
                    row = {"domain": key, "noise_type": nt, "rate": q, "method": m, "n_seeds": len(rs)}
                    for mt in ("accuracy", "violation_f1", "unseen_accuracy", "P_intended", "w_map", "n_refinements",
                               "first_refinement_t", "n_hypotheses_end"):
                        mu, lo, hi = boot([r.get(mt) for r in rs])
                        row.update({f"{mt}_mean": mu, f"{mt}_lo": lo, f"{mt}_hi": hi})
                    if m != "mlci":
                        row["refinement_rate"] = float(np.mean([(r["n_refinements"] or 0) > 0 for r in rs]))
                        row["map_intended_rate"] = float(np.mean([bool(r["map_is_intended"]) for r in rs]))
                        row["map_counts"] = dict(collections.Counter((r["map_description"] or "")[:80] for r in rs).most_common(3))
                    summ.append(row)
    write_csv(os.path.join(OUT, "summary.csv"), summ)
    # paired differences across seeds (same dataset): primary - fixed / oracle / mlci, and primary(q) - primary(0)
    paired = []
    for key in ("cart", "aisle"):
        for nt in NOISE_TYPES:
            for q in cfg["noise_rates"]:
                base = {r["seed"]: r for r in ex if r["domain"] == key and r["noise_type"] == nt and r["rate"] == q
                        and r["method"] == PRIMARY}
                for other in ("fixed", "oracle", "mlci", "adaptive_raw_rho0.1"):
                    o = {r["seed"]: r for r in ex if r["domain"] == key and r["noise_type"] == nt and r["rate"] == q
                         and r["method"] == other}
                    for mt in ("accuracy", "violation_f1", "unseen_accuracy"):
                        d = [base[s][mt] - o[s][mt] for s in base if s in o and base[s][mt] is not None and o[s][mt] is not None]
                        if d:
                            paired.append({"domain": key, "noise_type": nt, "rate": q, "comparison": f"{PRIMARY} - {other}",
                                           "metric": mt, "diff": fmt(boot(d)), "diff_mean": float(np.mean(d))})
                z = {r["seed"]: r for r in ex if r["domain"] == key and r["noise_type"] == nt and r["rate"] == 0.0
                     and r["method"] == PRIMARY}
                for mt in ("accuracy", "n_refinements"):
                    d = [base[s][mt] - z[s][mt] for s in base if s in z and base[s][mt] is not None]
                    if d and q > 0:
                        paired.append({"domain": key, "noise_type": nt, "rate": q, "comparison": f"{PRIMARY}: q - (q=0)",
                                       "metric": mt, "diff": fmt(boot(d)), "diff_mean": float(np.mean(d))})
    write_csv(os.path.join(OUT, "paired_differences.csv"), paired)
    # trigger response by demonstration category (fixed runs: unrefined hypothesis sets, all three statistics)
    cats = {"clean": lambda e: not e["corrupted"], "behavioural (compliant)": lambda e: e["kind"] == "behavioural_noise",
            "norm-violating": lambda e: e["violates_hidden_norm"]}
    resp = []
    for key in ("cart", "aisle"):
        fx = [e for e in ev if e["domain"] == key and e["method"] == "fixed" and e["rate"] > 0]
        for fam in ("raw", "task", "rel"):
            for cname, fn in cats.items():
                # per-seed mean of the within-run standardised score (z relative to the run's clean demos)
                per_seed = collections.defaultdict(list)
                runs = collections.defaultdict(list)
                for e in fx:
                    runs[(e["seed"], e["noise_type"], e["rate"])].append(e)
                for (s, nt, q), es in runs.items():
                    cl = [x[f"score_{fam}"] for x in es if not x["corrupted"]]
                    if len(cl) < 3:
                        continue
                    mu, sd = np.mean(cl), np.std(cl) + 1e-9
                    per_seed[s] += [(x[f"score_{fam}"] - mu) / sd for x in es if fn(x)]
                vals = [np.mean(v) for v in per_seed.values() if v]
                resp.append({"domain": key, "statistic": fam, "category": cname, "n_demos": sum(len(v) for v in per_seed.values()),
                             "mean_standardised_score": fmt(boot(vals)), "mean": boot(vals)[0]})
        # trigger firing rate by category for the adaptive variants (first refinement opportunities included)
        for m in ("adaptive_task_q0.1", "adaptive_rel_q0.1", "adaptive_raw_rho0.1"):
            es = [e for e in ev if e["domain"] == key and e["method"] == m and e["t"] >= 3]
            for cname, fn in cats.items():
                per_seed = collections.defaultdict(list)
                for e in es:
                    if fn(e):
                        per_seed[e["seed"]].append(float(e["triggered"]))
                vals = [np.mean(v) for v in per_seed.values() if v]
                resp.append({"domain": key, "statistic": f"{m} firing rate", "category": cname,
                             "n_demos": sum(len(v) for v in per_seed.values()), "mean_standardised_score": fmt(boot(vals)),
                             "mean": boot(vals)[0]})
    write_csv(os.path.join(OUT, "trigger_response_by_category.csv"), resp)
    write_json(os.path.join(OUT, "aggregate.json"), {"summary": summ, "paired": paired, "trigger_response": resp})
    figures(summ, ev, resp, cfg)
    for s in summ:
        if s["method"] in (PRIMARY, "fixed", "oracle", "mlci", "adaptive_raw_rho0.1"):
            print(f"{s['domain']:5s} {s['noise_type']:11s} q={s['rate']:.2f} {s['method']:20s} acc={fmt((s['accuracy_mean'], s['accuracy_lo'], s['accuracy_hi']))} "
                  f"F1={s['violation_f1_mean']:.2f} unseen={s['unseen_accuracy_mean'] if s['unseen_accuracy_mean'] is None else round(s['unseen_accuracy_mean'], 2)} "
                  f"nref={s['n_refinements_mean'] if s['n_refinements_mean'] is None else round(s['n_refinements_mean'], 2)} "
                  f"P={s['P_intended_mean'] if s['P_intended_mean'] is None else round(s['P_intended_mean'], 2)} "
                  f"w={s['w_map_mean'] if s['w_map_mean'] is None else round(s['w_map_mean'], 1)}")
    for r in resp:
        print(r)


def figures(summ, ev, resp, cfg):
    P.setup()
    rates = np.array(cfg["noise_rates"]) * 100
    dn = {"cart": "Exp. 1: cart-use", "aisle": "Exp. 2: aisle-use"}
    main_methods = [PRIMARY, "fixed", "oracle", "mlci"]

    def line(ax, key, nt, m, mt):
        ss = sorted([s for s in summ if s["domain"] == key and s["noise_type"] == nt and s["method"] == m],
                    key=lambda s: s["rate"])
        if not ss or ss[0][f"{mt}_mean"] is None:
            return
        P.mean_ci_line(ax, [s["rate"] * 100 for s in ss], [s[f"{mt}_mean"] for s in ss], [s[f"{mt}_lo"] for s in ss],
                       [s[f"{mt}_hi"] for s in ss], STYLE[m], label=STYLE[m]["label"])

    for nt, tag in (("behavioural", "behavioral"), ("violation", "violation")):
        for mt, fname in (("accuracy", "accuracy"), ("violation_f1", "f1")):
            fig, axes = P.plt.subplots(1, 2, figsize=(10, 3.5))
            for ax, key in zip(axes, ("cart", "aisle")):
                for m in main_methods:
                    line(ax, key, nt, m, mt)
                ax.set_ylim(-0.03, 1.03); ax.set_title(dn[key])
                ax.set_xlabel(f"{'behavioral (compliant)' if nt == 'behavioural' else 'norm-violation'} noise (% of 20 demos)")
                ax.set_ylabel("held-out accuracy" if mt == "accuracy" else "violation F1")
            axes[1].legend(fontsize=6.5, loc="lower left")
            P.save(fig, os.path.join(FIG, f"{tag}_{fname}.png"))
    fig, axes = P.plt.subplots(1, 2, figsize=(10, 3.5))
    for ax, nt in zip(axes, NOISE_TYPES):
        for m in main_methods:
            line(ax, "aisle", nt, m, "unseen_accuracy")
        ax.set_ylim(-0.03, 1.03); ax.set_xlabel(f"{nt} noise (%)"); ax.set_ylabel("unseen-aisle accuracy")
        ax.set_title(f"Exp. 2 unseen aisle – {nt} noise")
    axes[0].legend(fontsize=6.5, loc="lower left")
    P.save(fig, os.path.join(FIG, "aisle_unseen_accuracy.png"))
    fig, axes = P.plt.subplots(1, 2, figsize=(10, 3.5), sharey=True)
    for ax, key in zip(axes, ("cart", "aisle")):
        for nt, ls in (("behavioural", "-"), ("violation", "--")):
            for m in (PRIMARY, "adaptive_rel_q0.1", "adaptive_raw_rho0.1"):
                ss = sorted([s for s in summ if s["domain"] == key and s["noise_type"] == nt and s["method"] == m],
                            key=lambda s: s["rate"])
                ax.plot([s["rate"] * 100 for s in ss], [s["n_refinements_mean"] for s in ss], color=STYLE[m]["color"],
                        marker=STYLE[m]["marker"], ls=ls, ms=4, label=f"{STYLE[m]['label']} – {nt}")
        ax.set_xlabel("noise (% of 20 demos)"); ax.set_ylabel("mean refinement triggers per run"); ax.set_title(dn[key])
    axes[0].legend(fontsize=5.5)
    P.save(fig, os.path.join(FIG, "refinements_vs_noise.png"))
    fig, axes = P.plt.subplots(1, 2, figsize=(10, 3.6))
    cats = ["clean", "behavioural (compliant)", "norm-violating"]
    for ax, key in zip(axes, ("cart", "aisle")):
        for i, fam in enumerate(("raw", "task", "rel")):
            vals = [next((r["mean"] for r in resp if r["domain"] == key and r["statistic"] == fam and r["category"] == c), np.nan)
                    for c in cats]
            ax.bar(np.arange(3) + (i - 1) * 0.26, vals, width=0.24, color=P.SLOTS[[6, 0, 4][i]],
                   label={"raw": "raw predictive", "task": "task-adjusted (selected)", "rel": "norm-relative"}[fam])
        ax.axhline(0, color=P.INK2, lw=0.8)
        ax.set_xticks(range(3)); ax.set_xticklabels(cats, fontsize=8)
        ax.set_ylabel("score, z-units vs the run's clean demos"); ax.set_title(dn[key] + " (fixed hypothesis sets)")
    axes[0].legend(fontsize=7)
    P.save(fig, os.path.join(FIG, "trigger_score_by_noise_type.png"))
    fig, axes = P.plt.subplots(1, 2, figsize=(10, 3.5))
    for ax, key in zip(axes, ("cart", "aisle")):
        for nt, ls in (("behavioural", "-"), ("violation", "--")):
            for m in (PRIMARY, "oracle", "fixed"):
                ss = sorted([s for s in summ if s["domain"] == key and s["noise_type"] == nt and s["method"] == m],
                            key=lambda s: s["rate"])
                ax.plot([s["rate"] * 100 for s in ss], [s["w_map_mean"] for s in ss], color=STYLE[m]["color"],
                        marker=STYLE[m]["marker"], ls=ls, ms=4, label=f"{STYLE[m]['label']} – {nt}")
        ax.set_xlabel("noise (% of 20 demos)"); ax.set_ylabel("posterior mean w of MAP hypothesis"); ax.set_title(dn[key])
    axes[0].legend(fontsize=5.5)
    P.save(fig, os.path.join(FIG, "inferred_w_vs_noise.png"))


if __name__ == "__main__":
    main()
