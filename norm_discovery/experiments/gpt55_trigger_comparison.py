"""Refinement-trigger comparison with actual GPT-5.5 proposals.

Only the trigger varies; proposal mechanism (GPT-5.5), abstraction language, Bayesian inference, priors,
demonstrations and held-out sets are unchanged.  Families (inference/triggers.py): raw, len, task, rel.

Protocols: 'orig' = original 3 demonstrations (10 GPT runs per domain; replicate keys orig/<domain>/<r>, so cached
responses from earlier GPT-5.5 experiments are reused when the prompt is identical); 'pool' = 20 clean
demonstrations from the task pool (same generator and seeds as the deterministic ablation's threshold pools;
replicate keys trig/<domain>/<r>).

Stages (each writes under results/gpt55/refinement_trigger_comparison/; nothing else is modified):
  paths   unrefined learner on every sequence: all four trigger statistics at every t, demo length, C_task,
          optimal task cost C*(context)                                            (GPT: initial proposals only)
  labels  counterfactual first refinement at every opportunity t >= 3: posterior with vs without the GPT-5.5
          refinement of D_1:t, at time t and at the end of the sequence.  Used ONLY for post-hoc evaluation
          (usefulness labels, precision/recall); never to choose thresholds.     (GPT: one refinement per (run, t))
  runs    full runs for every trigger family with the quantile rule q in {0.01, 0.05, 0.10, 0.20}, plus the paper's
          raw ratio rule (rho = 0.1) as reference; repeated refinements allowed.  (GPT: first refinements are cache
          hits from 'labels'; repeated refinements need new calls)
  analyze offline sweeps, confound statistics, precision/recall, figures, summary  (no API calls)

Every stage has a cache-only dry run that prints the number of new API calls:  <stage> --set dry=true
    bash -lc 'source ~/.bashrc && cd /home/train/norm_discovery && \
              /home/train/anaconda3/bin/python -m experiments.gpt55_trigger_comparison <stage> [--set dry=true]'
"""
from __future__ import annotations

import collections
import json
import os
import sys
import time

import numpy as np
from scipy.stats import spearmanr

from analysis import plotting as P
from environment.domains import get_spec
from environment.trajectories import plan_optimal
from experiments.common import (DEFAULTS, RESULTS, ROOT, bootstrap_ci, make_pool_dataset, make_provider, make_zc,
                                parse_args, pmap, run_metadata, seed_for, write_csv, write_json, write_jsonl)
from inference.triggers import FAMILIES, FAMILY_NAMES

NAME = "refinement_trigger_comparison"
QS = [0.01, 0.05, 0.10, 0.20]
USEFUL_DP = 0.05          # usefulness: Delta P(intended) >= 0.05, or any accuracy / unseen-accuracy gain


def rdir(*parts):
    d = os.path.join(RESULTS, "gpt55", NAME, *parts)
    os.makedirs(d, exist_ok=True)
    return d


def fdir():
    d = os.path.join(ROOT, "figures", "gpt55", NAME)
    os.makedirs(d, exist_ok=True)
    return d


def sequence(cfg, key, proto, r):
    spec = get_spec(key)
    if proto == "orig":
        return spec.training, f"orig/{key}/{r}"
    return make_pool_dataset(spec, 20, seed_for(cfg, "pool", key, r)), f"trig/{key}/{r}"


def provider(cfg, replicate, info):
    return make_provider(dict(cfg, openai_cache_only=bool(cfg.get("dry"))), replicate=replicate, run_info=info)


def _save_transcript(d, prov):
    comp = getattr(prov, "complete", None)
    write_json(os.path.join(d, "transcript.json"),
               {"llm_calls": [{k: v for k, v in c.items() if k != "output_text"} for c in getattr(comp, "calls", [])],
                "proposals": getattr(prov, "transcript", None)})


def _calls(prov):
    comp = getattr(prov, "complete", None)
    cs = getattr(comp, "calls", [])
    fresh = [c for c in cs if not c.get("cache_hit")]
    return {"n_calls": len(cs), "n_fresh": len(fresh),
            "input_tokens": sum((c.get("usage") or {}).get("input_tokens", 0) for c in fresh),
            "output_tokens": sum((c.get("usage") or {}).get("output_tokens", 0) for c in fresh),
            "models": sorted({c.get("returned_model") for c in cs if c.get("returned_model")})}


# ================================================================================================= stage: paths
def _path_job(job):
    from inference.refinement import LearnerConfig, NormLearner
    from proposals.openai_provider import CacheMissError
    cfg, key, proto, r = job
    spec = get_spec(key)
    demos, rep = sequence(cfg, key, proto, r)
    prov = provider(cfg, rep, {"experiment": NAME, "stage": "paths", "domain": key, "protocol": proto, "run_index": r})
    L = NormLearner(spec, prov, LearnerConfig(refinement=False), make_zc(spec, cfg))
    out = {"domain": key, "protocol": proto, "run_index": r, "error": None}
    try:
        res = L.run(demos, record_prefix_eval=True)
    except CacheMissError:
        return dict(out, miss=1)
    except Exception as e:
        return dict(out, error=f"{type(e).__name__}: {e}")
    steps = []
    for s, t in zip(res["trace"][1:], demos[1:]):
        cstar = len(plan_optimal(spec.domain, t.task, None).actions)
        steps.append({"t": s["t"], "demo_len": len(t.actions), "C_task": t.cost, "C_star": cstar,
                      "excess_cost": t.cost - cstar, **{f"score_{k}": v for k, v in s["scores"].items()},
                      "P_intended": s["P_intended"], "accuracy": s["accuracy"], "unseen_accuracy": s["unseen_accuracy"]})
    out.update({"steps": steps, "final": {k: res["final"].get(k) for k in ("accuracy", "violation_f1", "unseen_accuracy",
                                                                            "P_intended", "map_id", "map_description",
                                                                            "map_is_intended")},
                **_calls(prov)})
    if not cfg.get("dry"):
        d = rdir("paths", key, proto, f"run_{r:02d}")
        _save_transcript(d, prov)
        write_json(os.path.join(d, "metrics.json"), {k: v for k, v in out.items()})
    return out


# ================================================================================================= stage: labels
def _label_job(job):
    """Counterfactual: refine (once) at opportunity t; compare with not refining.  For evaluation only."""
    from inference.refinement import LearnerConfig, NormLearner
    from proposals.openai_provider import CacheMissError
    from experiments.gpt55_threshold_extended import _probe_batches, intended_abstraction_proposed
    cfg, key, proto, r, t = job
    spec = get_spec(key)
    demos, rep = sequence(cfg, key, proto, r)
    prov = provider(cfg, rep, {"experiment": NAME, "stage": "labels", "domain": key, "protocol": proto,
                               "run_index": r, "opportunity_t": t})
    L = NormLearner(spec, prov, LearnerConfig(refinement=False), make_zc(spec, cfg))
    out = {"domain": key, "protocol": proto, "run_index": r, "t": t, "error": None}
    try:
        L.post.add_demo(demos[0])
        L._accept(prov.propose_initial(spec, demos[:1]), 0)
        for tau in demos[1:t]:
            L.post.add_demo(tau)
        before = L._eval_now()
        n0 = len(L.post.hyps)
        L._refine(demos[:t])
        after = L._eval_now()
        new = [h.to_dict() for h in L.post.hyps[n0:]]
        for tau in demos[t:]:
            L.post.add_demo(tau)
        end = L._eval_now()
    except CacheMissError:
        return dict(out, miss=1)
    except Exception as e:
        return dict(out, error=f"{type(e).__name__}: {e}")
    abs_ok, concepts = intended_abstraction_proposed(spec, new, _probe_batches(spec))
    useful = ((after["P_intended"] - before["P_intended"] >= USEFUL_DP) or (after["accuracy"] > before["accuracy"])
              or ((after["unseen_accuracy"] or 0) > (before["unseen_accuracy"] or 0)))
    out.update({"before": before, "after": after, "end_if_refined_here": end, "useful": bool(useful),
                "intended_abstraction_proposed": abs_ok, "abstraction_concepts": concepts,
                "intended_norm_proposed": any(spec.is_intended(h) for h in L.post.hyps[n0:]),
                "n_new_hypotheses": len(new), **_calls(prov)})
    if not cfg.get("dry"):
        d = rdir("labels", key, proto, f"run_{r:02d}", f"t_{t:02d}")
        _save_transcript(d, prov)
        write_json(os.path.join(d, "metrics.json"), out)
    return out


# ================================================================================================= stage: runs
def configs():
    out = [{"family": f, "rule": "quantile", "param": q, "name": f"{f}_q{q}"} for f in FAMILIES for q in QS]
    out.append({"family": "raw", "rule": "ratio", "param": float(np.log(0.1)), "name": "raw_ratio_rho0.1 (paper)"})
    return out


def _run_job(job):
    from inference.refinement import LearnerConfig, NormLearner
    from proposals.openai_provider import CacheMissError
    from experiments.gpt55_threshold_extended import _probe_batches, intended_abstraction_proposed
    cfg, key, proto, r, c = job
    spec = get_spec(key)
    demos, rep = sequence(cfg, key, proto, r)
    prov = provider(cfg, rep, {"experiment": NAME, "stage": "runs", "domain": key, "protocol": proto, "run_index": r,
                               "trigger": c["name"]})
    lc = LearnerConfig(trigger_family=c["family"], trigger_rule=c["rule"], trigger_param=c["param"],
                       record_refinement_eval=True)
    L = NormLearner(spec, prov, lc, make_zc(spec, cfg))
    out = {"domain": key, "protocol": proto, "run_index": r, "trigger": c["name"], "family": c["family"],
           "rule": c["rule"], "param": c["param"], "error": None}
    try:
        res = L.run(demos, record_prefix_eval=True)
    except CacheMissError:
        return dict(out, miss=1)
    except Exception as e:
        return dict(out, error=f"{type(e).__name__}: {e}")
    f = res["final"]
    lvl1 = [h["json"] for h in f["all_hypotheses"] if h["level"] == 1]
    abs_ok = intended_abstraction_proposed(spec, lvl1, _probe_batches(spec))[0] if lvl1 else False
    steps = [{"t": s["t"], "demo_len": s["demo_len"], "C_task": s.get("C_task"), "trigger_score": s.get("trigger_score"),
              "baseline": s.get("trigger_baseline"), "threshold": s.get("trigger_threshold"), "triggered": s["triggered"],
              "n_refinements_so_far": s.get("n_refinements_so_far"), "refinement_eval": s.get("refinement_eval"),
              "P_intended": s["P_intended"], "accuracy": s.get("accuracy")} for s in res["trace"][1:]]
    out.update({"steps": steps, "refinement_points": f["refinement_points"], "n_refinements": f["n_refinements"],
                "accuracy": f["accuracy"], "violation_f1": f["violation_f1"], "unseen_accuracy": f["unseen_accuracy"],
                "P_intended": f["P_intended"] or 0.0, "map_id": f["map_id"], "map_description": f["map_description"],
                "map_is_intended": f["map_is_intended"], "intended_norm_proposed": f["intended_present"],
                "intended_abstraction_proposed": abs_ok, **_calls(prov)})
    if not cfg.get("dry"):
        d = rdir("runs", key, proto, c["name"].split(" ")[0], f"run_{r:02d}")
        _save_transcript(d, prov)
        write_json(os.path.join(d, "metrics.json"), out)
    return out


# ================================================================================================= drivers
def _jobs_paths(cfg):
    return [(cfg, k, p, r) for k in ("cart", "aisle") for p in ("orig", "pool") for r in range(cfg["gpt_runs"])]


def _jobs_labels(cfg):
    return [(cfg, k, p, r, t) for k in ("cart", "aisle") for p in ("orig", "pool") for r in range(cfg["gpt_runs"])
            for t in range(3, (3 if p == "orig" else 20) + 1)]


def _jobs_runs(cfg):
    return [(cfg, k, p, r, c) for k in ("cart", "aisle") for p in ("orig", "pool") for r in range(cfg["gpt_runs"])
            for c in configs()]


def stage(cfg, name, fn, jobs):
    t0 = time.time()
    res = pmap(fn, jobs, cfg["gpt_workers"])
    miss = sum(x.get("miss", 0) for x in res)
    errors = [x for x in res if x.get("error")]
    if cfg.get("dry"):
        print(json.dumps({"stage": name, "jobs": len(jobs), "jobs_needing_new_api_calls(first miss)": miss,
                          "note": "runs stage: each missing job may need >1 call (repeated refinements)"}, indent=1))
        return res
    write_jsonl(os.path.join(rdir(), f"{name}.jsonl"), res)
    tok = {"fresh_calls": sum(x.get("n_fresh", 0) for x in res),
           "input_tokens": sum(x.get("input_tokens", 0) for x in res),
           "output_tokens": sum(x.get("output_tokens", 0) for x in res),
           "models": sorted({m for x in res for m in x.get("models", [])}), "errors": len(errors),
           "error_examples": sorted({e["error"][:200] for e in errors})[:5], "runtime_s": round(time.time() - t0)}
    write_json(os.path.join(rdir(), f"{name}_api_usage.json"), tok)
    print(name, json.dumps(tok))
    return res


def _load(name):
    p = os.path.join(rdir(), f"{name}.jsonl")
    with open(p) as f:
        return [json.loads(l) for l in f]


# ================================================================================================= analysis
def _r2(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if np.std(x) == 0:
        return 0.0
    A = np.vstack([x, np.ones_like(x)]).T
    coef = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(1 - np.sum((y - A @ coef) ** 2) / np.sum((y - y.mean()) ** 2))


def first_trigger(scores, rule, param):
    """Offline first-trigger time on an unrefined path (scores for t = 2..T in order)."""
    from inference.triggers import should_trigger
    acc = []
    for t, s in scores:
        trig, _, _ = should_trigger(rule, param, s, acc)
        if trig:
            return t
        acc.append(s)
    return None


def analyze(cfg):
    paths = [p for p in _load("paths") if not p.get("error")]
    labels = {(l["domain"], l["protocol"], l["run_index"], l["t"]): l for l in _load("labels") if not l.get("error")}
    runs = [r for r in _load("runs") if not r.get("error")] if os.path.exists(os.path.join(rdir(), "runs.jsonl")) else []
    missing = {"paths": [(p["domain"], p["protocol"], p["run_index"]) for p in _load("paths") if p.get("error")],
               "labels": [(l["domain"], l["protocol"], l["run_index"], l["t"]) for l in _load("labels") if l.get("error")],
               "runs": [(r["domain"], r["protocol"], r["run_index"], r["trigger"]) for r in
                        (_load("runs") if runs else []) if r.get("error")]}
    od = rdir()
    # ---------------------------------------------------------------- 1. confound table (pool paths)
    conf = []
    for key in ("cart", "aisle"):
        st = [s for p in paths if p["domain"] == key and p["protocol"] == "pool" for s in p["steps"]]
        for f in FAMILIES:
            y = [s[f"score_{f}"] for s in st]
            row = {"domain": key, "family": f, "n": len(y)}
            for x in ("demo_len", "C_task", "C_star", "excess_cost"):
                xv = [s[x] for s in st]
                row[f"spearman_{x}"] = float(spearmanr(xv, y).correlation)
                row[f"R2_{x}"] = _r2(xv, y)
            conf.append(row)
    write_csv(os.path.join(od, "confound_table.csv"), conf)
    # ---------------------------------------------------------------- 2. per-step trigger log (paths)
    step_rows = [{"domain": p["domain"], "protocol": p["protocol"], "run_index": p["run_index"], **s} for p in paths
                 for s in p["steps"]]
    write_csv(os.path.join(od, "path_scores.csv"), step_rows)
    # ---------------------------------------------------------------- 3. offline sweeps (first refinement)
    q_grid = [0.0, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.75, 1.0]
    sweep = []
    for key in ("cart", "aisle"):
        for proto in ("orig", "pool"):
            ps = [p for p in paths if p["domain"] == key and p["protocol"] == proto]
            for f in FAMILIES:
                # margins for the ratio rule: quantiles of the pooled (s_t - running min) differences, per domain
                diffs = []
                for p in ps:
                    acc = []
                    for s in p["steps"]:
                        if acc:
                            diffs.append(s[f"score_{f}"] - min(acc))
                        acc.append(s[f"score_{f}"])
                for rule, grid in (("quantile", q_grid), ("ratio", sorted(set(np.round(np.quantile(diffs, np.linspace(0, 1, 21)) + 1e-9, 6))) if diffs else [])):
                    for param in grid:
                        tp = fp = fn = tn = 0
                        n_ref, end_acc, end_unseen, end_P, firsts = 0, [], [], [], []
                        for p in ps:
                            sc = [(s["t"], s[f"score_{f}"]) for s in p["steps"]]
                            t1 = first_trigger(sc, rule, param)
                            for (t, _) in sc:
                                if t < 3:
                                    continue
                                if t1 is not None and t > t1:
                                    break
                                lab = labels.get((key, proto, p["run_index"], t))
                                if lab is None:
                                    continue
                                fired = t == t1
                                if fired and lab["useful"]:
                                    tp += 1
                                elif fired:
                                    fp += 1
                                elif lab["useful"]:
                                    fn += 1
                                else:
                                    tn += 1
                            if t1 is not None and (key, proto, p["run_index"], t1) in labels:
                                n_ref += 1
                                firsts.append(t1)
                                e = labels[(key, proto, p["run_index"], t1)]["end_if_refined_here"]
                            else:
                                e = p["final"]
                            end_acc.append(e["accuracy"])
                            end_unseen.append(e.get("unseen_accuracy"))
                            end_P.append(e.get("P_intended") or 0.0)
                        sweep.append({"domain": key, "protocol": proto, "family": f, "rule": rule, "param": float(param),
                                      "n_runs": len(ps), "first_refinement_rate": n_ref / max(len(ps), 1),
                                      "median_first_t": float(np.median(firsts)) if firsts else None,
                                      "TP": tp, "FP": fp, "FN": fn, "TN": tn,
                                      "precision": tp / (tp + fp) if tp + fp else None,
                                      "recall": tp / (tp + fn) if tp + fn else None,
                                      "FPR": fp / (fp + tn) if fp + tn else None, "FNR": fn / (fn + tp) if fn + tp else None,
                                      "single_refinement_accuracy": float(np.mean(end_acc)) if end_acc else None,
                                      "single_refinement_unseen": (float(np.mean([u for u in end_unseen if u is not None]))
                                                                   if any(u is not None for u in end_unseen) else None),
                                      "single_refinement_P_intended": float(np.mean(end_P)) if end_P else None})
    write_csv(os.path.join(od, "offline_sweep.csv"), sweep)
    # ---------------------------------------------------------------- 4. full GPT runs: summary + repeated refinements
    summ = []
    for key in ("cart", "aisle"):
        for proto in ("orig", "pool"):
            for c in configs():
                rs = [r for r in runs if r["domain"] == key and r["protocol"] == proto and r["trigger"] == c["name"]]
                if not rs:
                    continue
                complete = len(rs) == cfg["gpt_runs"]
                row = {"domain": key, "protocol": proto, "trigger": c["name"], "family": c["family"], "rule": c["rule"],
                       "param": c["param"], "n_runs": len(rs), "complete": complete}
                for m in ("accuracy", "violation_f1", "unseen_accuracy", "P_intended", "n_refinements"):
                    v = [r[m] for r in rs if r.get(m) is not None]
                    if v:
                        mu, lo, hi = bootstrap_ci(v, cfg["bootstrap"])
                        row.update({f"{m}_mean": mu, f"{m}_sd": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                                    f"{m}_ci_lo": lo, f"{m}_ci_hi": hi})
                row["refinement_rate"] = float(np.mean([r["n_refinements"] > 0 for r in rs]))
                refd = [r for r in rs if r["n_refinements"] > 0]
                row["intended_abstraction_rate_among_refined"] = (float(np.mean([r["intended_abstraction_proposed"]
                                                                                 for r in refd])) if refd else None)
                row["intended_norm_rate_among_refined"] = (float(np.mean([r["intended_norm_proposed"] for r in refd]))
                                                           if refd else None)
                # usefulness of every refinement event: first via counterfactual label, repeats via before/after
                useful_first, repeats, useless_repeats = [], 0, 0
                for r in rs:
                    for i, t in enumerate(r["refinement_points"]):
                        if i == 0:
                            lab = labels.get((key, proto, r["run_index"], t))
                            if lab is not None:
                                useful_first.append(lab["useful"])
                        else:
                            repeats += 1
                            ev = next(s["refinement_eval"] for s in r["steps"] if s["t"] == t)
                            b, a = ev["before"], ev["after"]
                            u = ((a["P_intended"] - b["P_intended"] >= USEFUL_DP) or a["accuracy"] > b["accuracy"]
                                 or (a["unseen_accuracy"] or 0) > (b["unseen_accuracy"] or 0))
                            useless_repeats += (not u)
                row["first_refinement_precision"] = float(np.mean(useful_first)) if useful_first else None
                row["repeated_refinements_per_run"] = repeats / len(rs)
                row["unnecessary_repeated_refinements_per_run"] = useless_repeats / len(rs)
                summ.append(row)
    write_csv(os.path.join(od, "summary.csv"), summ)
    # ---------------------------------------------------------------- 5. scale stability across domains
    scale = []
    for f in FAMILIES:
        row = {"family": f}
        for key in ("cart", "aisle"):
            d = [s for s in sweep if s["domain"] == key and s["protocol"] == "pool" and s["family"] == f and s["rule"] == "ratio"]
            m50 = next((s["param"] for s in sorted(d, key=lambda s: s["param"]) if s["first_refinement_rate"] >= 0.5), None)
            row[f"{key}_margin_for_50pct_first_refinement"] = m50
            st = [x[f"score_{f}"] for p in paths if p["domain"] == key and p["protocol"] == "pool" for x in p["steps"]]
            row[f"{key}_score_median"] = float(np.median(st)) if st else None
            row[f"{key}_score_IQR"] = float(np.subtract(*np.percentile(st, [75, 25]))) if st else None
        scale.append(row)
    write_csv(os.path.join(od, "scale_stability.csv"), scale)
    labels_tab = collections.Counter((l["domain"], l["protocol"], l["useful"]) for l in labels.values())
    agg = {"families": FAMILY_NAMES, "quantiles": QS, "usefulness_rule": f"dP(intended) >= {USEFUL_DP} or accuracy gain "
           "or unseen-accuracy gain, refined vs not refined at the same t (evaluation only)",
           "label_counts": {f"{k[0]}/{k[1]}/{'useful' if k[2] else 'not useful'}": v for k, v in labels_tab.items()},
           "confound_table": conf, "scale_stability": scale, "summary": summ, "missing": missing}
    write_json(os.path.join(od, "aggregate.json"), agg)
    figures(paths, sweep, summ, runs)
    for r in conf:
        print(f"confound {r['domain']:5s} {r['family']:4s} rho(len)={r['spearman_demo_len']:+.2f} R2(len)={r['R2_demo_len']:.2f} "
              f"rho(C*)={r['spearman_C_star']:+.2f} R2(C*)={r['R2_C_star']:.2f} rho(excess)={r['spearman_excess_cost']:+.2f}")
    for s in summ:
        print(f"{s['domain']:5s} {s['protocol']:4s} {s['trigger']:26s} ref={s['refinement_rate']:.2f} "
              f"nref={s.get('n_refinements_mean', 0):.2f} acc={s.get('accuracy_mean', float('nan')):.3f} "
              f"unseen={s.get('unseen_accuracy_mean', float('nan')):.3f} P={s.get('P_intended_mean', float('nan')):.3f} "
              f"prec1={s['first_refinement_precision']} rep={s['repeated_refinements_per_run']:.2f} "
              f"useless_rep={s['unnecessary_repeated_refinements_per_run']:.2f} complete={s['complete']}")
    print(json.dumps(agg["label_counts"]), json.dumps(scale, indent=0))


def figures(paths, sweep, summ, runs):
    P.setup()
    fd = fdir()
    col = {f: P.SLOTS[i] for i, f in enumerate(FAMILIES)}
    mk = {"raw": "o", "len": "s", "task": "^", "rel": "D"}
    dname = {"cart": "Exp. 1: cart-use", "aisle": "Exp. 2: aisle-use"}
    # Fig 1: statistic vs trajectory length
    fig, axes = P.plt.subplots(2, 4, figsize=(15, 6.2))
    for i, key in enumerate(("cart", "aisle")):
        st = [s for p in paths if p["domain"] == key and p["protocol"] == "pool" for s in p["steps"]]
        for j, f in enumerate(FAMILIES):
            ax = axes[i][j]
            x = [s["demo_len"] for s in st]
            y = [s[f"score_{f}"] for s in st]
            ax.scatter(x, y, s=9, color=col[f], alpha=0.55, edgecolor="none")
            rr = spearmanr(x, y).correlation if len(x) > 2 else float("nan")
            ax.set_title(f"{dname[key]} – {FAMILY_NAMES[f]}\nSpearman ρ = {rr:+.2f}, R² = {_r2(x, y):.2f}", fontsize=8)
            ax.set_xlabel("demonstration length (= C_task)")
            ax.set_ylabel("trigger statistic")
    P.save(fig, os.path.join(fd, "fig1_statistic_vs_length.png"))
    # Fig 2: refinement rate vs threshold (quantile rule, offline sweep, pool + orig)
    fig, axes = P.plt.subplots(1, 2, figsize=(10, 3.4))
    for ax, key in zip(axes, ("cart", "aisle")):
        for f in FAMILIES:
            for proto, ls in (("pool", "-"), ("orig", ":")):
                d = sorted([s for s in sweep if s["domain"] == key and s["protocol"] == proto and s["family"] == f
                            and s["rule"] == "quantile"], key=lambda s: s["param"])
                ax.plot([s["param"] for s in d], [s["first_refinement_rate"] for s in d], color=col[f], marker=mk[f],
                        ls=ls, ms=4, label=f"{FAMILY_NAMES[f]} ({proto})" if key == "cart" else None)
        ax.set_xlabel("quantile threshold q"); ax.set_ylabel("fraction of runs that refine"); ax.set_title(dname[key])
        ax.set_ylim(-0.03, 1.03)
    axes[0].legend(fontsize=6, ncol=1)
    P.save(fig, os.path.join(fd, "fig2_refinement_rate_vs_threshold.png"))
    # Fig 3: held-out accuracy vs refinement rate (offline single-refinement sweep, pool) + full GPT runs
    fig, axes = P.plt.subplots(1, 2, figsize=(10, 3.4))
    for ax, key in zip(axes, ("cart", "aisle")):
        for f in FAMILIES:
            d = [s for s in sweep if s["domain"] == key and s["protocol"] == "pool" and s["family"] == f]
            ax.scatter([s["first_refinement_rate"] for s in d], [s["single_refinement_accuracy"] for s in d], color=col[f],
                       marker=mk[f], s=18, alpha=0.6, label=FAMILY_NAMES[f] + " (sweep)")
            g = [s for s in summ if s["domain"] == key and s["protocol"] == "pool" and s["family"] == f and s["rule"] == "quantile"]
            ax.scatter([s["refinement_rate"] for s in g], [s.get("accuracy_mean") for s in g], color=col[f], marker=mk[f],
                       s=60, edgecolor=P.INK, lw=0.8, label=FAMILY_NAMES[f] + " (full GPT runs)")
        ax.set_xlabel("fraction of runs that refine"); ax.set_ylabel("held-out accuracy (end of sequence)")
        ax.set_title(dname[key] + " – 20-demo sequences"); ax.set_ylim(-0.03, 1.03)
    axes[1].legend(fontsize=5.5)
    P.save(fig, os.path.join(fd, "fig3_accuracy_vs_refinement_rate.png"))
    # Fig 4: unseen-aisle accuracy vs threshold
    fig, ax = P.plt.subplots(figsize=(6.2, 3.4))
    for f in FAMILIES:
        d = sorted([s for s in sweep if s["domain"] == "aisle" and s["protocol"] == "pool" and s["family"] == f
                    and s["rule"] == "quantile"], key=lambda s: s["param"])
        ax.plot([s["param"] for s in d], [s["single_refinement_unseen"] for s in d], color=col[f], marker=mk[f], ms=4,
                label=FAMILY_NAMES[f])
        g = sorted([s for s in summ if s["domain"] == "aisle" and s["protocol"] == "pool" and s["family"] == f
                    and s["rule"] == "quantile"], key=lambda s: s["param"])
        ax.scatter([s["param"] for s in g], [s.get("unseen_accuracy_mean") for s in g], color=col[f], marker=mk[f], s=60,
                   edgecolor=P.INK, lw=0.8)
    ax.set_xlabel("quantile threshold q (line: single-refinement sweep; large markers: full GPT runs)")
    ax.set_ylabel("unseen-aisle accuracy"); ax.set_ylim(-0.03, 1.03); ax.legend(fontsize=7)
    P.save(fig, os.path.join(fd, "fig4_unseen_accuracy_vs_threshold.png"))
    # Fig 5: precision / recall
    fig, axes = P.plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, key in zip(axes, ("cart", "aisle")):
        for f in FAMILIES:
            d = [s for s in sweep if s["domain"] == key and s["protocol"] == "pool" and s["family"] == f
                 and s["precision"] is not None and s["recall"] is not None]
            d = sorted(d, key=lambda s: s["recall"])
            ax.plot([s["recall"] for s in d], [s["precision"] for s in d], color=col[f], marker=mk[f], ms=4, lw=1,
                    label=FAMILY_NAMES[f])
        ax.set_xlabel("recall (useful refinement opportunities)"); ax.set_ylabel("precision")
        ax.set_xlim(-0.03, 1.03); ax.set_ylim(-0.03, 1.03); ax.set_title(dname[key] + " – 20-demo sequences")
    axes[0].legend(fontsize=7)
    P.save(fig, os.path.join(fd, "fig5_precision_recall.png"))
    # Fig 6: number of refinements per 20-demo sequence (full GPT runs)
    if runs:
        fig, axes = P.plt.subplots(1, 2, figsize=(11, 3.6))
        for ax, key in zip(axes, ("cart", "aisle")):
            labels_, data = [], []
            for c in configs():
                v = [r["n_refinements"] for r in runs if r["domain"] == key and r["protocol"] == "pool" and r["trigger"] == c["name"]]
                if v:
                    labels_.append(c["name"].replace("_ratio_rho0.1 (paper)", " ρ=0.1").replace("_q", " q="))
                    data.append(v)
            ax.boxplot(data, vert=True, widths=0.6)
            ax.set_xticks(range(1, len(labels_) + 1)); ax.set_xticklabels(labels_, rotation=70, fontsize=6.5)
            ax.set_ylabel("refinements per 20-demo sequence"); ax.set_title(dname[key])
        P.save(fig, os.path.join(fd, "fig6_refinements_per_sequence.png"))


def main():
    stages = ("paths", "labels", "runs", "analyze")
    if len(sys.argv) < 2 or sys.argv[1] not in stages:
        sys.exit(f"usage: python -m experiments.gpt55_trigger_comparison {{{','.join(stages)}}} [--set dry=true]")
    st = sys.argv.pop(1)
    DEFAULTS.update({"gpt_workers": 8, "dry": False})
    cfg = parse_args(f"trigger comparison: {st}")
    cfg["provider"] = "openai"
    if st != "analyze" and not cfg["dry"] and not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY is not set: run `source ~/.bashrc` first (no fallback provider).")
    if not cfg["dry"] and st != "analyze":
        write_json(os.path.join(rdir(), f"metadata_{st}.json"), run_metadata(f"{NAME}_{st}", cfg))
    t0 = time.time()
    if st == "paths":
        stage(cfg, "paths", _path_job, _jobs_paths(cfg))
    elif st == "labels":
        stage(cfg, "labels", _label_job, _jobs_labels(cfg))
    elif st == "runs":
        stage(cfg, "runs", _run_job, _jobs_runs(cfg))
    else:
        analyze(cfg)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
