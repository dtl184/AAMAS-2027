"""Extended GPT-5.5 refinement-threshold study (original 3-demonstration protocol, rho beyond 1).

Characterises the CURRENT algorithm unchanged: trigger p_t < rho * b_t with the running-minimum baseline.
Writes only to results/gpt55/refinement_threshold_extended/ and figures/gpt55/refinement_threshold_extended/;
existing results (results/gpt55/*, results/deterministic_proposals/*) are read but never modified.  New GPT-5.5
responses are added to results/llm_cache/ (existing cache entries are never rewritten).

    bash -lc 'source ~/.bashrc && cd /home/train/norm_discovery && \
              /home/train/anaconda3/bin/python -m experiments.gpt55_threshold_extended dry-run'
    bash -lc 'source ~/.bashrc && cd /home/train/norm_discovery && \
              /home/train/anaconda3/bin/python -m experiments.gpt55_threshold_extended run'
"""
from __future__ import annotations

import collections
import json
import os
import sys
import time

import numpy as np

from analysis import plotting as P
from environment.domains import get_spec
from experiments.common import (DEFAULTS, RESULTS, ROOT, bootstrap_ci, make_provider, make_zc, parse_args, pmap,
                                run_metadata, write_csv, write_json, write_jsonl)

NAME = "refinement_threshold_extended"
BASE_RHOS = [0.1, 0.5, 1, 2, 5, 10, 20, 40, 80, 100, 500, 1000, 5000, 10000, 20000]
# breakpoints added from the empirical trigger ratios of the original GPT-5.5 runs
# (cart r_3 in [35.84, 36.93]; aisle r_3 in [13720, 13810])
EXTRA_RHOS = [35, 36, 37, 50, 13000, 13750, 14000, 15000]
RHOS = sorted(set(BASE_RHOS + EXTRA_RHOS))


def res_dir():
    d = os.path.join(RESULTS, "gpt55", NAME)
    os.makedirs(d, exist_ok=True)
    return d


def fig_dir():
    d = os.path.join(ROOT, "figures", "gpt55", NAME)
    os.makedirs(d, exist_ok=True)
    return d


# ------------------------------------------------------------------------------------------ trigger ratios
def trigger_ratios():
    rows = []
    with open(os.path.join(RESULTS, "gpt55", "original", "runs.jsonl")) as f:
        for l in f:
            r = json.loads(l)
            if r["method"] != "full" or r.get("error"):
                continue
            for s in r["trace"]:
                if s.get("log_ratio_p_over_b") is not None:
                    rows.append({"domain": r["domain"], "run_index": r["run_index"], "t": s["t"],
                                 "log_p": s["log_p"], "log_b": s["log_b"],
                                 "ratio_p_over_b": float(np.exp(s["log_ratio_p_over_b"])),
                                 "source": "results/gpt55/original (full method, rho = 0.1)"})
    stats = {}
    for d in ("cart", "aisle"):
        v = np.array([r["ratio_p_over_b"] for r in rows if r["domain"] == d])
        if len(v):
            stats[d] = {"n": int(len(v)), "min": float(v.min()), "p25": float(np.percentile(v, 25)),
                        "median": float(np.median(v)), "mean": float(v.mean()), "p75": float(np.percentile(v, 75)),
                        "max": float(v.max())}
    return rows, stats


# ------------------------------------------------------------------------------------------ concept checks
REF_CONCEPTS = {
    "cart": {"pickup": [("event", {"pre": {"and": ["INTERACT", {"not": "hasCart"}]}, "post": "hasCart"})],
             "return": [("event", {"pre": {"and": ["INTERACT", "hasCart", "at(0,1)"]}, "post": {"not": "hasCart"}}),
                        ("prop", "cartAt(0,1)")]},
    "aisle": {"aisle": [("prop", {"or": [{"and": ["shelfW", "shelfE"]}, {"and": ["shelfN", "shelfS"]}]})]},
}


def _probe_batches(spec):
    from norms.violations import trajectory_batch
    return [trajectory_batch(spec.domain, t) for t in spec.probe_set()]


def _concept_vector(abstraction, name, batches):
    return np.concatenate([abstraction.eval(name, b, "pre") for b in batches])


def intended_abstraction_proposed(spec, hyp_dicts, batches):
    """True iff the refined hypotheses jointly define every key concept of the intended abstraction
    (behavioural equivalence of the defined property/event on all probe-set transitions)."""
    from abstractions.base import Abstraction
    refs = {}
    for concept, defs in REF_CONCEPTS[spec.key].items():
        vecs = []
        for kind, f in defs:
            a = Abstraction(events={"_r": (f["pre"], f["post"])}) if kind == "event" else Abstraction(props={"_r": f})
            vecs.append(_concept_vector(a, "_r", batches))
        refs[concept] = vecs
    found = {c: False for c in refs}
    for hd in hyp_dicts:
        if hd.get("level") != 1:
            continue
        from norms.hypotheses import Hypothesis
        a = Hypothesis.from_dict(hd).abstraction
        for name in list(a.props) + list(a.events):
            try:
                v = _concept_vector(a, name, batches)
            except Exception:
                continue
            for c, vecs in refs.items():
                if any(np.array_equal(v, rv) for rv in vecs):
                    found[c] = True
    return all(found.values()), found


# ------------------------------------------------------------------------------------------ jobs
def _dry_job(job):
    """Run with the cache only; report whether a new API call would be needed and the prompt size."""
    from inference.refinement import LearnerConfig, NormLearner
    from proposals.openai_provider import CacheMissError
    cfg, key, r, rho = job
    spec = get_spec(key)
    prov = make_provider(dict(cfg, openai_cache_only=True), replicate=f"orig/{key}/{r}")
    L = NormLearner(spec, prov, LearnerConfig(rho=rho), make_zc(spec, cfg))
    try:
        L.run(spec.training)
        return {"domain": key, "run_index": r, "rho": rho, "miss": False}
    except CacheMissError as e:
        return {"domain": key, "run_index": r, "rho": rho, "miss": True, "prompt_chars": len(e.prompt),
                "prompt_hash": hash(e.prompt)}


def _run_job(job):
    from experiments.gpt55 import _job
    cfg, key, r, rho = job
    return _job({"experiment": NAME, "domain": key, "rep": r, "method": "full", "cfg": cfg, "demos": "original",
                 "rho": rho, "cond": f"rho_{rho}", "replicate": f"orig/{key}/{r}"})


def dry_run(cfg):
    jobs = [(cfg, key, r, rho) for key in ("cart", "aisle") for r in range(cfg["gpt_runs"]) for rho in RHOS]
    res = pmap(_dry_job, jobs, cfg["workers"])
    misses = {(x["domain"], x["run_index"], x["prompt_hash"]): x for x in res if x["miss"]}   # identical prompts -> 1 call
    n = len(misses)
    tok_in = sum(m["prompt_chars"] for m in misses.values()) / 4.0
    # output tokens: mean of the refinement calls already in the cache (fall back to the smoke-test value)
    outs = []
    for fn in os.listdir(os.path.join(RESULTS, "llm_cache")):
        if fn.endswith(".json"):
            with open(os.path.join(RESULTS, "llm_cache", fn)) as f:
                rec = json.load(f)
            outs.append((rec.get("usage") or {}).get("output_tokens", 0))
    out_tok = float(np.mean(outs)) if outs else 3622.0
    est = {"rhos": RHOS, "runs_total": len(jobs), "runs_fully_cached": len(jobs) - sum(x["miss"] for x in res),
           "new_api_calls_expected": n,
           "new_calls_by_domain": dict(collections.Counter(k[0] for k in misses)),
           "approx_input_tokens": int(tok_in), "approx_output_tokens": int(n * out_tok),
           "output_tokens_per_call_assumed": round(out_tok)}
    write_json(os.path.join(res_dir(), "dry_run_estimate.json"), est)
    print(json.dumps(est, indent=1))
    return est


def run(cfg):
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY is not set: run `source ~/.bashrc` first (no fallback provider is used).")
    od = res_dir()
    write_json(os.path.join(od, "metadata.json"), run_metadata("gpt55_" + NAME, cfg))
    tr_rows, tr_stats = trigger_ratios()
    write_csv(os.path.join(od, "trigger_ratios.csv"), tr_rows)
    jobs = [(cfg, key, r, rho) for key in ("cart", "aisle") for rho in RHOS for r in range(cfg["gpt_runs"])]
    rows = pmap(_run_job, jobs, cfg["gpt_workers"])
    # concept-level check of the refined abstraction
    batches = {k: _probe_batches(get_spec(k)) for k in ("cart", "aisle")}
    for r in rows:
        if r.get("error") or not r.get("n_refinements"):
            r["intended_abstraction_proposed"] = None if r.get("error") else False
            r["abstraction_concepts_found"] = None
            continue
        ok, found = intended_abstraction_proposed(get_spec(r["domain"]), [h["json"] for h in r["all_hypotheses"]],
                                                  batches[r["domain"]])
        r["intended_abstraction_proposed"], r["abstraction_concepts_found"] = ok, found
        tr = {s["t"]: s for s in r["trace"]}
        r["n_hypotheses_before_refinement"] = tr[r["refinement_points"][0] - 1]["n_hypotheses"]
        r["n_hypotheses_after_refinement"] = tr[r["refinement_points"][0]]["n_hypotheses"]
    write_jsonl(os.path.join(od, "runs.jsonl"), rows)
    per_step = []
    for r in rows:
        for s in (r.get("trace") or []):
            if s.get("log_p") is None:
                continue
            per_step.append({"domain": r["domain"], "run_index": r["run_index"], "rho": r["rho"], "t": s["t"],
                             "p_t": float(np.exp(s["log_p"])), "log_p_t": s["log_p"],
                             "b_t": None if s["log_b"] is None else float(np.exp(s["log_b"])), "log_b_t": s["log_b"],
                             "p_over_b": None if s["log_ratio_p_over_b"] is None else float(np.exp(s["log_ratio_p_over_b"])),
                             "rho_times_b": None if s.get("log_rho_b") is None else float(np.exp(s["log_rho_b"])),
                             "triggered": s["triggered"], "n_hypotheses": s["n_hypotheses"]})
    write_csv(os.path.join(od, "per_step.csv"), per_step)
    summarise_and_plot(rows, tr_rows, tr_stats, cfg)


def summarise_and_plot(rows, tr_rows, tr_stats, cfg):
    od = res_dir()
    summ, missing = [], []
    for key in ("cart", "aisle"):
        for rho in RHOS:
            rs = [r for r in rows if r["domain"] == key and r["rho"] == rho]
            ok = [r for r in rs if not r.get("error")]
            bad = [r for r in rs if r.get("error")]
            missing += [{"domain": key, "rho": rho, "run_index": r["run_index"], "error": r["error"][:300]} for r in bad]
            complete = len(ok) == cfg["gpt_runs"]
            row = {"domain": key, "rho": rho, "n_runs_ok": len(ok), "n_runs_failed": len(bad), "complete": complete}
            if complete:
                for m in ("accuracy", "violation_f1", "unseen_accuracy", "P_intended", "n_refinements",
                          "n_hypotheses"):
                    v = [(r.get(m) or 0.0) if m == "P_intended" else r.get(m) for r in ok]
                    v = [x for x in v if x is not None]
                    if v:
                        mu, lo, hi = bootstrap_ci(v, cfg["bootstrap"])
                        row.update({f"{m}_mean": mu, f"{m}_sd": float(np.std(v, ddof=1)), f"{m}_ci_lo": lo,
                                    f"{m}_ci_hi": hi})
                refd = [r for r in ok if (r.get("n_refinements") or 0) > 0]
                row["refinement_rate"] = len(refd) / len(ok)
                row["refinement_points"] = sorted({p for r in refd for p in r["refinement_points"]})
                row["n_refined"] = len(refd)
                row["intended_abstraction_rate_among_refined"] = (
                    float(np.mean([bool(r["intended_abstraction_proposed"]) for r in refd])) if refd else None)
                row["intended_norm_rate_among_refined"] = (
                    float(np.mean([bool(r.get("intended_present")) for r in refd])) if refd else None)
                row["map_intended_rate"] = float(np.mean([bool(r.get("map_is_intended")) for r in ok]))
                row["map_counts"] = dict(collections.Counter((r.get("map_description") or "")[:100] for r in ok))
                row["mean_hyps_before_refinement"] = (float(np.mean([r["n_hypotheses_before_refinement"] for r in refd]))
                                                      if refd else None)
                row["mean_hyps_after_refinement"] = (float(np.mean([r["n_hypotheses_after_refinement"] for r in refd]))
                                                     if refd else None)
            summ.append(row)
    write_csv(os.path.join(od, "summary.csv"), summ)
    write_csv(os.path.join(od, "missing_runs.csv"), missing)
    agg = {"rhos": RHOS, "trigger_ratio_stats_from_original_runs": tr_stats, "summary": summ,
           "missing_runs": missing,
           "api": {"returned_models": sorted({m for r in rows for m in (r.get("returned_models") or [])}),
                   "fresh_calls": int(sum(r.get("n_llm_calls_fresh") or 0 for r in rows)),
                   "input_tokens_fresh": int(sum(r.get("input_tokens") or 0 for r in rows)),
                   "output_tokens_fresh": int(sum(r.get("output_tokens") or 0 for r in rows))}}
    write_json(os.path.join(od, "aggregate.json"), agg)
    plots(summ, tr_rows)
    for s in summ:
        print(f"{s['domain']:5s} rho={s['rho']:<8} complete={s['complete']} refrate={s.get('refinement_rate')} "
              f"acc={s.get('accuracy_mean', float('nan')):.3f} unseen={s.get('unseen_accuracy_mean', float('nan')):.3f} "
              f"P_int={s.get('P_intended_mean', float('nan')):.3f} abs={s.get('intended_abstraction_rate_among_refined')} "
              f"norm={s.get('intended_norm_rate_among_refined')} failed={s['n_runs_failed']}")
    print(json.dumps(agg["api"]))


def plots(summ, tr_rows):
    P.setup()
    fd = fig_dir()
    title = {"cart": "Exp. 1: cart-use", "aisle": "Exp. 2: aisle-use"}
    col = {"cart": P.SLOTS[0], "aisle": P.SLOTS[1]}
    mk = {"cart": "o", "aisle": "s"}

    def series(key, m):
        ss = [s for s in summ if s["domain"] == key and s["complete"]]
        return [s["rho"] for s in ss], ss

    def boundary(ax):
        for key in ("cart", "aisle"):
            v = [r["ratio_p_over_b"] for r in tr_rows if r["domain"] == key]
            if v:
                ax.axvspan(min(v), max(v), color=col[key], alpha=0.15, lw=0)
                ax.axvline(np.median(v), color=col[key], ls=":", lw=1)

    # Plot 1: refinement rate
    fig, ax = P.plt.subplots(figsize=(6.2, 3.4))
    for key in ("cart", "aisle"):
        x, ss = series(key, "refinement_rate")
        ax.plot(x, [s["refinement_rate"] for s in ss], color=col[key], marker=mk[key], label=title[key])
    boundary(ax)
    ax.set_xscale("log"); ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("refinement threshold ρ (log scale)"); ax.set_ylabel("fraction of GPT-5.5 runs that refine")
    ax.legend(fontsize=7); ax.set_title("Refinement rate vs ρ (shaded: observed p₃/b₃ range)")
    P.save(fig, os.path.join(fd, "refinement_rate_vs_rho.png"))
    # Plot 2: accuracy (+ unseen for aisle)
    fig, axes = P.plt.subplots(1, 2, figsize=(10, 3.4))
    for ax, key in zip(axes, ("cart", "aisle")):
        x, ss = series(key, "accuracy")
        P.mean_ci_line(ax, x, [s["accuracy_mean"] for s in ss], [s["accuracy_ci_lo"] for s in ss],
                       [s["accuracy_ci_hi"] for s in ss], dict(P.METHOD_STYLE["full"], color=col[key]),
                       label="held-out accuracy")
        if key == "aisle":
            P.mean_ci_line(ax, x, [s["unseen_accuracy_mean"] for s in ss], [s["unseen_accuracy_ci_lo"] for s in ss],
                           [s["unseen_accuracy_ci_hi"] for s in ss], dict(P.METHOD_STYLE["oracle"], color=P.SLOTS[6]),
                           label="unseen-aisle accuracy")
        boundary(ax)
        ax.set_xscale("log"); ax.set_ylim(-0.03, 1.03)
        ax.set_xlabel("ρ (log scale)"); ax.set_ylabel("accuracy"); ax.set_title(title[key]); ax.legend(fontsize=7)
    P.save(fig, os.path.join(fd, "accuracy_vs_rho.png"))
    # Plot 3: intended-abstraction / intended-norm proposal rate among refined runs
    fig, ax = P.plt.subplots(figsize=(6.2, 3.4))
    for key in ("cart", "aisle"):
        ss = [s for s in summ if s["domain"] == key and s["complete"] and s.get("n_refined")]
        x = [s["rho"] for s in ss]
        ax.plot(x, [s["intended_abstraction_rate_among_refined"] for s in ss], color=col[key], marker=mk[key],
                label=f"{title[key]}: intended abstraction")
        ax.plot(x, [s["intended_norm_rate_among_refined"] for s in ss], color=col[key], marker=mk[key], ls="--",
                mfc="white", label=f"{title[key]}: intended norm")
    boundary(ax)
    ax.set_xscale("log"); ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("ρ (log scale)"); ax.set_ylabel("rate among runs that refined")
    ax.set_title("Intended abstraction / norm proposed after refinement"); ax.legend(fontsize=6.5)
    P.save(fig, os.path.join(fd, "intended_proposal_rate_vs_rho.png"))
    # Plot 4: P(intended)
    fig, ax = P.plt.subplots(figsize=(6.2, 3.4))
    for key in ("cart", "aisle"):
        x, ss = series(key, "P_intended")
        ax.errorbar(x, [s["P_intended_mean"] for s in ss], yerr=[s["P_intended_sd"] for s in ss], color=col[key],
                    marker=mk[key], capsize=3, label=f"{title[key]} (mean ± SD)")
    boundary(ax)
    ax.set_xscale("log"); ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("ρ (log scale)"); ax.set_ylabel("P(intended | D)"); ax.legend(fontsize=7)
    ax.set_title("Posterior of the intended norm vs ρ")
    P.save(fig, os.path.join(fd, "p_intended_vs_rho.png"))
    # Plot 5: trigger-ratio distribution with tested rho values
    fig, ax = P.plt.subplots(figsize=(7, 2.8))
    for i, key in enumerate(("cart", "aisle")):
        v = [r["ratio_p_over_b"] for r in tr_rows if r["domain"] == key]
        ax.scatter(v, np.full(len(v), i) + np.random.default_rng(0).uniform(-0.12, 0.12, len(v)), color=col[key],
                   marker=mk[key], s=22, label=f"{title[key]}: observed p₃/b₃ (10 runs)")
    for rho in RHOS:
        ax.axvline(rho, color=P.INK2, lw=0.6, alpha=0.6)
    ax.set_xscale("log"); ax.set_yticks([0, 1]); ax.set_yticklabels(["cart", "aisle"])
    ax.set_xlabel("p_t / b_t (log scale); grey lines = tested ρ values"); ax.set_ylim(-0.6, 1.6)
    ax.legend(fontsize=7, loc="upper left"); ax.set_title("Trigger ratios of the original GPT-5.5 runs")
    P.save(fig, os.path.join(fd, "trigger_ratio_distribution.png"))


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("dry-run", "run", "summarise"):
        sys.exit("usage: python -m experiments.gpt55_threshold_extended {dry-run,run,summarise}")
    cmd = sys.argv.pop(1)
    DEFAULTS.update({"gpt_workers": 8})
    cfg = parse_args("Extended GPT-5.5 refinement-threshold study")
    cfg["provider"] = "openai"
    cfg["_args"]["out"] = None
    t0 = time.time()
    if cmd == "dry-run":
        tr_rows, tr_stats = trigger_ratios()
        print("trigger ratio stats (original GPT-5.5 runs):", json.dumps(tr_stats, indent=1))
        dry_run(cfg)
    elif cmd == "run":
        run(cfg)
    else:   # re-aggregate from runs.jsonl without any API call
        with open(os.path.join(res_dir(), "runs.jsonl")) as f:
            rows = [json.loads(l) for l in f]
        tr_rows, tr_stats = trigger_ratios()
        summarise_and_plot(rows, tr_rows, tr_stats, cfg)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
