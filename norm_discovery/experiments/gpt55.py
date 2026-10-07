"""Experiments with ACTUAL GPT-5.5 hypothesis proposal (OpenAI API).  Results go to results/gpt55/, figures to
figures/gpt55/; they are never mixed with the deterministic-proposal ablation (results/deterministic_proposals/).

Always run in a shell where the API key is loaded:
    bash -lc 'source ~/.bashrc && cd /home/train/norm_discovery && PY=/home/train/anaconda3/bin/python && \
              $PY -m experiments.gpt55 <command> [--set key=value ...]'

Commands
  estimate         dry run (no API calls): expected number of initial / refinement calls and prompt tokens
  original         original protocol (3 demos, rho = 0.1), gpt_runs independent GPT runs per domain;
                   full / fixed / oracle with GPT proposals, MLCI (no LLM) for reference
  threshold        rho grid on the original protocol, gpt_runs GPT runs per rho and domain
  noise            noise experiment (20 demos), gpt_noise_seeds GPT runs per condition
  identifiability  (A) proposal quality over gpt_runs runs; (B) Bayesian identifiability of the diagnostic
                   demonstration given each GPT run's own hypothesis set (no extra calls)

Independent runs: run r uses replicate key "<experiment-family>/<domain>/<r>" in the response cache, so no two
runs share a response; the same run re-issuing an identical prompt (e.g. at another rho) reuses its own response.
A run that hits an API / parse / model-mismatch error is stopped and recorded (field "error"), never re-routed to
another provider or model.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time

import numpy as np

from analysis import plotting as P
from environment.domains import get_spec
from experiments.common import (DEFAULTS, bootstrap_ci, fig_dir, make_pool_dataset, out_dir, parse_args, pmap,
                                run_metadata, run_method, seed_for, write_csv, write_json, write_jsonl)

METRIC_KEYS = ["accuracy", "violation_f1", "unseen_accuracy", "precision", "recall", "map_id", "map_description",
               "map_level", "map_is_intended", "map_is_intended_exact", "P_intended", "P_intended_exact",
               "intended_present", "intended_proposed_at_level", "map_posterior", "n_refinements",
               "refinement_points", "n_hypotheses", "n_hypotheses_generated", "n_rejected_proposals",
               "posterior_entropy", "runtime_s", "final_level", "error", "n_constraints", "constraints", "predictions"]


def _require_openai(cfg):
    cfg["provider"] = "openai"
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY is not set: run `source ~/.bashrc` first (no fallback provider is used).")


def _usage(result):
    calls = result.get("llm_calls") or []
    fresh = [c for c in calls if not c.get("cache_hit")]
    tok_in = sum((c.get("usage") or {}).get("input_tokens", 0) for c in fresh)
    tok_out = sum((c.get("usage") or {}).get("output_tokens", 0) for c in fresh)
    models = sorted({c.get("returned_model") for c in calls if c.get("returned_model")})
    return {"n_llm_calls": len(calls), "n_llm_calls_fresh": len(fresh), "input_tokens": tok_in,
            "output_tokens": tok_out, "returned_models": models,
            "n_retries": sum(len(c.get("retries") or []) for c in calls)}


def _job(job):
    """job = dict(experiment, domain, rep, method, cfg, rho, demos=('original'|('pool', seed, type, rate)), cond)"""
    cfg, key = job["cfg"], job["domain"]
    spec = get_spec(key)
    if job["demos"] == "original":
        demos = spec.training
    else:
        _, seed, ntype, rate = job["demos"]
        demos = make_pool_dataset(spec, cfg["noise_demos"], seed, ntype, rate)
    info = {"experiment": job["experiment"], "domain": key, "run_index": job["rep"], "condition": job.get("cond")}
    r = run_method(spec, job["method"], demos, cfg, rho=job.get("rho"), replicate=job["replicate"],
                   run_info=info, record_posteriors=(job["method"] != "mlci"), record_prefix_eval=True)
    r.update({"experiment": job["experiment"], "domain": key, "run_index": job["rep"], "condition": job.get("cond"),
              "rho": job.get("rho") if job["method"] != "mlci" else None, "replicate_key": job["replicate"],
              **(_usage(r) if job["method"] != "mlci" else {})})
    if job["demos"] != "original":
        r["corrupted_positions"] = [i + 1 for i, t in enumerate(demos) if t.meta.get("corrupted")]
        r["noise_type"], r["rate"] = job["demos"][2] or "none", job["demos"][3]
    _persist(job, r, cfg)
    return {k: v for k, v in r.items() if k not in ("transcript",)}


def _persist(job, r, cfg):
    parts = [out_dir(cfg, job["experiment"]), job["domain"]]
    if job.get("cond"):
        parts.append(job["cond"])
    parts.append(f"{job['method']}_run_{job['rep']:02d}" if isinstance(job["rep"], int) else f"{job['method']}")
    d = os.path.join(*parts)
    os.makedirs(d, exist_ok=True)
    hyper = {k: cfg[k] for k in ("beta", "lam", "gamma", "w_max", "w_step", "prior_mode", "dl_coding",
                                 "openai_model", "openai_temperature", "openai_reasoning_effort",
                                 "llm_n_hypotheses", "llm_max_parse_retries")}
    hyper["rho"] = job.get("rho") if job.get("rho") is not None else cfg["rho"]
    hyper["W"] = list(np.round(np.arange(0, cfg["w_max"] + 1e-9, cfg["w_step"]), 6))
    write_json(os.path.join(d, "metrics.json"), {"hyperparameters": hyper, "replicate_key": job["replicate"],
                                                 **{k: r.get(k) for k in METRIC_KEYS},
                                                 **{k: r.get(k) for k in ("n_llm_calls", "n_llm_calls_fresh",
                                                                          "input_tokens", "output_tokens",
                                                                          "returned_models", "n_retries")},
                                                 "all_hypotheses": r.get("all_hypotheses"),
                                                 "posterior_table": r.get("posterior_table")})
    write_json(os.path.join(d, "trace.json"), r.get("trace"))
    if r.get("transcript") is not None:
        write_json(os.path.join(d, "transcript.json"), {"llm_calls": r.get("llm_calls"),
                                                        "proposals": r["transcript"]})


def _agg(rows, keys, n_boot):
    out = []
    groups = collections.OrderedDict()
    for r in rows:
        groups.setdefault(tuple(r.get(k) for k in keys), []).append(r)
    for g, rs in groups.items():
        ok = [r for r in rs if not r.get("error")]
        row = dict(zip(keys, g))
        row.update({"n_runs": len(rs), "n_errors": len(rs) - len(ok),
                    "errors": sorted({r["error"][:200] for r in rs if r.get("error")})})
        for m in ("accuracy", "violation_f1", "unseen_accuracy", "P_intended", "n_refinements", "n_hypotheses",
                  "n_rejected_proposals", "input_tokens", "output_tokens"):
            vals = [(r.get(m) or 0.0) if m == "P_intended" else r.get(m) for r in ok]
            vals = [v for v in vals if v is not None]
            if vals and not (m == "P_intended" and rs[0]["method"] == "mlci"):
                mu, lo, hi = bootstrap_ci(vals, n_boot)
                row.update({f"{m}_mean": mu, f"{m}_sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                            f"{m}_ci_lo": lo, f"{m}_ci_hi": hi})
        if ok and rs[0]["method"] != "mlci":
            row["refinement_rate"] = float(np.mean([(r.get("n_refinements") or 0) > 0 for r in ok]))
            fr = [r["refinement_points"][0] for r in ok if r.get("refinement_points")]
            row["refinement_index_values"] = fr
            row["intended_proposed_rate"] = float(np.mean([bool(r.get("intended_present")) for r in ok]))
            row["map_intended_rate"] = float(np.mean([bool(r.get("map_is_intended")) for r in ok]))
            row["map_counts"] = dict(collections.Counter(r.get("map_description", "")[:90] for r in ok))
        out.append(row)
    return out


# ============================================================================================ estimate
def cmd_estimate(cfg):
    """Dry run with the deterministic provider standing in for GPT: counts the calls each experiment would make
    and the size of every prompt the LLM provider would send (no API call is made)."""
    from inference.refinement import NormLearner, method_config
    from proposals.deterministic import DeterministicProposalProvider
    from proposals.llm_stub import LLMProposalProvider
    from experiments.common import make_zc, learner_kwargs

    class Counting(DeterministicProposalProvider):
        def __init__(self):
            self.prompts = []
            self.fmt = LLMProposalProvider(lambda p: "[]", n_hypotheses=cfg["llm_n_hypotheses"])

        def propose_initial(self, spec, demos):
            self.prompts.append(("initial", self.fmt._prompt(spec, demos[:1], 0, False)))
            return super().propose_initial(spec, demos)

        def refine(self, spec, demos, current, level):
            self.prompts.append(("refinement", self.fmt._prompt(spec, demos, level, True, current)))
            return super().refine(spec, demos, current, level)

    def count(spec, demos, method, rho=None):
        kw = learner_kwargs(cfg)
        if rho is not None:
            kw["rho"] = rho
        prov = Counting()
        NormLearner(spec, prov, method_config(method, **kw), make_zc(spec, cfg)).run(demos)
        return prov.prompts

    est = {}
    out_tok = 3622  # measured in the smoke test (incl. reasoning tokens)
    for exp in ("original", "noise"):
        calls = collections.Counter()
        tokens = 0
        for key in cfg["domains"]:
            spec = get_spec(key)
            if exp == "original":
                sets = [(spec.training, m, None) for m in ("full", "oracle")]
                mult = cfg["gpt_runs"]
            else:
                sets = []
                for s in range(cfg["gpt_noise_seeds"]):
                    for nt, q in [(None, 0.0)] + [(nt, q) for nt in ("behavioural", "violation")
                                                  for q in cfg["noise_rates"] if q > 0]:
                        demos = make_pool_dataset(spec, cfg["noise_demos"], seed_for(cfg, "noise", key, s), nt, q)
                        sets += [(demos, "full", None), (demos[:1], "oracle", None)]
                mult = 1
            for demos, m, rho in sets:
                for kind, p in count(spec, demos, m, rho):
                    if m == "oracle" and kind == "initial":
                        continue   # identical to the full method's initial prompt -> cached
                    calls[kind] += mult
                    tokens += mult * len(p) / 4.0
        n = sum(calls.values())
        est[exp] = {"calls_by_kind": dict(calls), "total_calls_upper_bound": n,
                    "approx_input_tokens": int(tokens), "approx_output_tokens": n * out_tok,
                    "note": "refinement counts use the deterministic provider's trigger behaviour as a proxy; GPT "
                            "hypothesis sets can trigger more or fewer refinements; input tokens ~ chars/4"}
    est["threshold"] = {"note": "re-uses the original runs' cached prompts; new calls only for runs that refine at "
                                "some rho but not at rho=0.1: at most gpt_runs x 2 domains refinement calls"}
    write_json(os.path.join(out_dir(cfg, "estimate"), "estimate.json"), est)
    print(json.dumps(est, indent=1))


# ============================================================================================ original
def cmd_original(cfg):
    _require_openai(cfg)
    od = out_dir(cfg, "original")
    write_json(os.path.join(od, "metadata.json"), run_metadata("gpt55_original", cfg))
    jobs = []
    for key in cfg["domains"]:
        for r in range(cfg["gpt_runs"]):
            for m in ("full", "fixed", "oracle"):
                jobs.append({"experiment": "original", "domain": key, "rep": r, "method": m, "cfg": cfg,
                             "demos": "original", "replicate": f"orig/{key}/{r}"})
        jobs.append({"experiment": "original", "domain": key, "rep": 0, "method": "mlci", "cfg": cfg,
                     "demos": "original", "replicate": "none"})
    rows = pmap(_job, jobs, cfg["gpt_workers"])
    write_jsonl(os.path.join(od, "runs.jsonl"), rows)
    summ = _agg(rows, ["domain", "method"], cfg["bootstrap"])
    write_csv(os.path.join(od, "summary.csv"), summ)
    _original_table(rows, summ, od)


def _original_table(rows, summ, od):
    det_dir = os.path.join(os.path.dirname(os.path.dirname(od)), "deterministic_proposals", "reconstruction")
    L = ["# GPT-5.5 original-protocol reproduction (3 demonstrations, rho = 0.1)", "",
         "Mean ± SD over independent GPT-5.5 runs (MLCI is deterministic and uses no LLM).", "",
         "| Method | Exp.1 Acc. | Exp.1 Viol. F1 | Exp.2 Acc. | Exp.2 Viol. F1 | Exp.2 Unseen Acc. |", "|---|---|---|---|---|---|"]
    name = {"fixed": "Fixed abstraction (GPT-5.5 initial proposals)", "mlci": "MLCI (no LLM)",
            "full": "Ours – GPT-5.5", "oracle": "Oracle abstraction (GPT-5.5 proposals)"}
    def cell(d, m, k):
        r = next((s for s in summ if s["domain"] == d and s["method"] == m), None)
        if r is None or f"{k}_mean" not in r:
            return "–"
        return f"{r[k + '_mean']:.2f} ± {r[k + '_sd']:.2f}" if r["n_runs"] > 1 else f"{r[k + '_mean']:.2f}"
    for m in ("fixed", "mlci", "full", "oracle"):
        L.append(f"| {name[m]} | {cell('cart', m, 'accuracy')} | {cell('cart', m, 'violation_f1')} | "
                 f"{cell('aisle', m, 'accuracy')} | {cell('aisle', m, 'violation_f1')} | {cell('aisle', m, 'unseen_accuracy')} |")
    L += ["", "| Domain | Method | runs (errors) | refined | refinement at t | intended proposed | final MAP = intended | "
              "mean P(intended) | accepted / rejected proposals (mean) |", "|---|---|---|---|---|---|---|---|---|"]
    for s in summ:
        if s["method"] == "mlci":
            continue
        rs = [r for r in rows if r["domain"] == s["domain"] and r["method"] == s["method"] and not r.get("error")]
        n = len(rs)
        acc_n = np.mean([r.get("n_hypotheses_generated") or 0 for r in rs]) if rs else 0
        L.append(f"| {s['domain']} | {s['method']} | {s['n_runs']} ({s['n_errors']}) | "
                 f"{sum((r.get('n_refinements') or 0) > 0 for r in rs)}/{n} | {s.get('refinement_index_values')} | "
                 f"{sum(bool(r.get('intended_present')) for r in rs)}/{n} | {sum(bool(r.get('map_is_intended')) for r in rs)}/{n} | "
                 f"{s.get('P_intended_mean', 0):.3f} | {acc_n:.1f} / {s.get('n_rejected_proposals_mean', 0):.1f} |")
    L += ["", "Paper (Table I): GPT-5.5 Exp.1 0.80/0.80, Exp.2 0.96/1.00/0.93; Table II: GPT-5.5 proposed 10/10 and "
              "10/10, final MAP 0/10 and 8/10.", ""]
    L += ["## Refinement trigger per run (full method)", "", "| Domain | run | log p2 | log p3 | log(ρ·b3) | p3/b3 | triggered |",
          "|---|---|---|---|---|---|---|"]
    for r in rows:
        if r["method"] != "full" or r.get("error"):
            continue
        tr = {s["t"]: s for s in r["trace"]}
        t3 = tr.get(3, {})
        L.append(f"| {r['domain']} | {r['run_index']} | {tr[2]['log_p']:.2f} | {t3.get('log_p', float('nan')):.2f} | "
                 f"{t3.get('log_rho_b') or float('nan'):.2f} | {np.exp(t3.get('log_ratio_p_over_b') or np.nan):.3g} | "
                 f"{t3.get('triggered')} |")
    with open(os.path.join(od, "TABLE.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


# ============================================================================================ threshold
def cmd_threshold(cfg):
    _require_openai(cfg)
    od = out_dir(cfg, "threshold_sensitivity")
    write_json(os.path.join(od, "metadata.json"), run_metadata("gpt55_threshold", cfg))
    jobs = [{"experiment": "threshold_sensitivity", "domain": key, "rep": r, "method": "full", "cfg": cfg,
             "demos": "original", "rho": rho, "cond": f"rho_{rho}", "replicate": f"orig/{key}/{r}"}
            for key in cfg["domains"] for rho in cfg["rho_grid"] for r in range(cfg["gpt_runs"])]
    rows = pmap(_job, jobs, cfg["gpt_workers"])
    write_jsonl(os.path.join(od, "runs.jsonl"), rows)
    summ = _agg(rows, ["domain", "method", "rho"], cfg["bootstrap"])
    write_csv(os.path.join(od, "summary.csv"), summ)
    P.setup()
    metrics = [("accuracy", "Held-out accuracy"), ("violation_f1", "Violation F1"), ("refinement_rate", "Refinement rate"),
               ("P_intended", "P(intended)")]
    fig, axes = P.plt.subplots(len(metrics), len(cfg["domains"]), figsize=(4.6 * len(cfg["domains"]), 2.7 * len(metrics)),
                               squeeze=False)
    for i, (m, lab) in enumerate(metrics):
        for j, key in enumerate(cfg["domains"]):
            ax = axes[i][j]
            ss = sorted([s for s in summ if s["domain"] == key], key=lambda s: s["rho"])
            x = [s["rho"] for s in ss]
            if m == "refinement_rate":
                ax.plot(x, [s.get(m, np.nan) for s in ss], color=P.SLOTS[0], marker="o")
            else:
                P.mean_ci_line(ax, x, [s.get(f"{m}_mean", np.nan) for s in ss], [s.get(f"{m}_ci_lo", np.nan) for s in ss],
                               [s.get(f"{m}_ci_hi", np.nan) for s in ss], P.METHOD_STYLE["full"], label="GPT-5.5 full method")
            ax.axvline(0.1, color=P.INK2, ls=":", lw=0.8)
            ax.set_xscale("log")
            ax.set_ylim(-0.03, 1.03)
            ax.set_xlabel("refinement threshold ρ")
            ax.set_ylabel(lab)
            ax.set_title({"cart": "Exp. 1: cart-use", "aisle": "Exp. 2: aisle-use"}[key])
    P.save(fig, os.path.join(fig_dir(cfg), "threshold_gpt55.png"))
    for s in summ:
        print(f"{s['domain']:5s} rho={s['rho']:<6} acc={s.get('accuracy_mean', float('nan')):.3f} "
              f"refrate={s.get('refinement_rate', float('nan')):.2f} P_int={s.get('P_intended_mean', float('nan')):.3f} "
              f"proposed={s.get('intended_proposed_rate')} map={s.get('map_intended_rate')} errors={s['n_errors']}")


# ============================================================================================ noise
def cmd_noise(cfg):
    _require_openai(cfg)
    od = out_dir(cfg, "noise_robustness")
    write_json(os.path.join(od, "metadata.json"), run_metadata("gpt55_noise", cfg))
    jobs = []
    for key in cfg["domains"]:
        for s in range(cfg["gpt_noise_seeds"]):
            seed = seed_for(cfg, "noise", key, s)     # identical datasets to the deterministic ablation
            conds = [(None, 0.0)] + [(nt, q) for nt in ("behavioural", "violation") for q in cfg["noise_rates"] if q > 0]
            for nt, q in conds:
                for m in cfg["gpt_noise_methods"]:
                    jobs.append({"experiment": "noise_robustness", "domain": key, "rep": s, "method": m, "cfg": cfg,
                                 "demos": ("pool", seed, nt, q), "cond": f"{nt or 'none'}_{q}",
                                 "replicate": f"noise/{key}/{s}" if m != "mlci" else "none"})
    rows = pmap(_job, jobs, cfg["gpt_workers"])
    write_jsonl(os.path.join(od, "runs.jsonl"), rows)
    expanded = []
    for r in rows:   # the q = 0 condition belongs to both noise-type curves
        if r["noise_type"] == "none":
            for nt in ("behavioural", "violation"):
                expanded.append(dict(r, noise_type=nt))
        else:
            expanded.append(r)
    summ = _agg(expanded, ["domain", "noise_type", "rate", "method"], cfg["bootstrap"])
    write_csv(os.path.join(od, "summary.csv"), summ)
    P.setup()
    lab = {"full": "GPT-5.5 full method", "fixed": "Fixed abstraction (GPT-5.5)", "oracle": "Oracle abstraction (GPT-5.5)",
           "mlci": "MLCI (no LLM)"}
    for nt in ("behavioural", "violation"):
        metrics = [("accuracy", "Held-out accuracy"), ("violation_f1", "Violation F1"), ("unseen_accuracy", "Unseen-aisle accuracy")]
        fig, axes = P.plt.subplots(len(metrics), len(cfg["domains"]), figsize=(4.6 * len(cfg["domains"]), 2.9 * len(metrics)),
                                   squeeze=False)
        for i, (m, ylab) in enumerate(metrics):
            for j, key in enumerate(cfg["domains"]):
                ax = axes[i][j]
                if m == "unseen_accuracy" and key == "cart":
                    ax.axis("off")
                    continue
                for meth in cfg["gpt_noise_methods"]:
                    ss = sorted([s for s in summ if s["domain"] == key and s["noise_type"] == nt and s["method"] == meth],
                                key=lambda s: s["rate"])
                    if not ss or any(f"{m}_mean" not in s for s in ss):
                        continue
                    P.mean_ci_line(ax, np.array([s["rate"] for s in ss]) * 100, [s[f"{m}_mean"] for s in ss],
                                   [s[f"{m}_ci_lo"] for s in ss], [s[f"{m}_ci_hi"] for s in ss], P.METHOD_STYLE[meth],
                                   label=lab[meth])
                ax.set_ylim(-0.03, 1.03)
                ax.set_xlabel(f"{nt} noise (% of demonstrations)")
                ax.set_ylabel(ylab)
                ax.set_title({"cart": "Exp. 1: cart-use", "aisle": "Exp. 2: aisle-use"}[key])
        axes[0][-1].legend(fontsize=7, loc="lower left")
        P.save(fig, os.path.join(fig_dir(cfg), f"noise_{nt}_gpt55.png"))
    for s in summ:
        print(f"{s['domain']:5s} {s['noise_type']:11s} q={s['rate']:.2f} {s['method']:6s} acc={s.get('accuracy_mean', float('nan')):.3f} "
              f"f1={s.get('violation_f1_mean', float('nan')):.3f} unseen={s.get('unseen_accuracy_mean', float('nan')):.3f} "
              f"P_int={s.get('P_intended_mean', float('nan')):.3f} errors={s['n_errors']}")


# ============================================================================================ identifiability
COMPETITORS = {"cart": ["H_int", "A0_any_to_last", "A0_first_to_last", "A0_interact_last", "A0_visit_last", "H_item",
                        "H_uncond", "H_visit", "H_leave", "H_noexit"],
               "aisle": ["S_inAisle", "S_corridor", "S_nearShelf", "S_noShelfInteractWithCart", "S_memorised",
                         "B0_cells", "B0_pickcells"]}


def _reference_hyps(spec):
    from norms.hypotheses import Hypothesis
    from proposals import templates as T
    if spec.key == "cart":
        ds = T.exp1_level0((1, 5), (0, 1)) + T.exp1_level1((0, 1))
    else:
        ds = T.exp2_level0([(3, 1), (3, 2)], [(3, 1)]) + T.exp2_level1([(3, 1), (3, 2), (6, 1), (6, 2)])
    return {d["id"]: Hypothesis.from_dict(d) for d in ds}


def _ident_proposals(job):
    """(A) one GPT run: initial proposals from D1, then refinement proposals from D1-D3 (forced, no trigger)."""
    from inference.refinement import LearnerConfig, NormLearner
    from experiments.common import make_provider, make_zc
    cfg, key, r = job["cfg"], job["domain"], job["rep"]
    spec = get_spec(key)
    prov = make_provider(cfg, replicate=f"ident/{key}/{r}",
                         run_info={"experiment": "identifiability", "domain": key, "run_index": r})
    L = NormLearner(spec, prov, LearnerConfig(), make_zc(spec, cfg))
    out = {"domain": key, "run_index": r, "error": None}
    try:
        L.post.add_demo(spec.training[0])
        L._accept(prov.propose_initial(spec, spec.training[:1]), 0)
        for t in spec.training[1:]:
            L.post.add_demo(t)
        L._refine(spec.training)
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
    refs = _reference_hyps(spec)
    found = {name: [h.hid for h in L.post.hyps if spec.equivalent(h, ref)] for name, ref in refs.items()}
    out.update({"n_hypotheses": len(L.post.hyps), "n_rejected": L.n_rejected,
                "equivalent_found": {k: v for k, v in found.items()},
                "proposed": {k: bool(v) for k, v in found.items()},
                "hypotheses": [h.to_dict() for h in L.post.hyps]})
    d = os.path.join(out_dir(cfg, "identifiability"), "proposals", key, f"run_{r:02d}")
    os.makedirs(d, exist_ok=True)
    comp = getattr(prov, "complete", None)
    write_json(os.path.join(d, "transcript.json"), {"llm_calls": [{k: v for k, v in c.items() if k != "output_text"}
                                                                  for c in getattr(comp, "calls", [])],
                                                    "proposals": prov.transcript})
    write_json(os.path.join(d, "metrics.json"), {k: v for k, v in out.items()})
    return out


def _ident_bayes(res, cfg):
    """(B) posterior along the diagnostic / control / reacquire sequences using this run's GPT hypothesis set."""
    from experiments import identifiability as I
    from inference.posterior import Posterior
    from inference.predictive import classify, evaluate_predictions
    from norms.hypotheses import Hypothesis
    from experiments.common import make_zc
    spec = get_spec("cart")
    d = spec.domain
    hyps = [Hypothesis.from_dict(h) for h in res["hypotheses"]]
    ordinary = [I.true_demo(spec, d.make_task(it), f"D{4 + k}") for k, it in enumerate(I.ORDINARY)]
    base5 = list(spec.training) + ordinary[:2]
    diag = I.true_demo(spec, d.make_task(("bread",), (d.entrance_idx, d.layout.cell_index[(3, 1)], 0, 0)), "D6")
    seqs = {"diagnostic": base5 + [diag, ordinary[2], ordinary[3]], "control": list(spec.training) + ordinary,
            "reacquire": base5 + [I.reacquire_demo(spec, ("apple", "milk"), "D6r"), ordinary[2], ordinary[3]]}
    out = []
    for lab, seq in seqs.items():
        post = Posterior(make_zc(spec, cfg), cfg["lam"], cfg["gamma"], cfg["prior_mode"], cfg["dl_coding"])
        for t in seq:
            post.add_demo(t)
        post.add_hypotheses(hyps)
        eq = [i for i, h in enumerate(post.hyps) if spec.is_intended(h)]
        for t in range(len(seq) + 1):
            m = post.marginal(upto=t)
            imap = int(np.argmax(m))
            ev = evaluate_predictions(spec, classify(d, post.hyps[imap], spec.heldout))
            order = [i for i in np.argsort(-m) if i not in eq]
            out.append({"run_index": res["run_index"], "sequence": lab, "t": t,
                        "P_intended": float(sum(m[i] for i in eq)), "intended_in_set": bool(eq),
                        "top_competitor": post.hyps[order[0]].describe() if order else None,
                        "P_top_competitor": float(m[order[0]]) if order else None,
                        "map": post.hyps[imap].describe(), "accuracy": ev["accuracy"]})
    return out


def cmd_identifiability(cfg):
    _require_openai(cfg)
    od = out_dir(cfg, "identifiability")
    write_json(os.path.join(od, "metadata.json"), run_metadata("gpt55_identifiability", cfg))
    jobs = [{"cfg": cfg, "domain": key, "rep": r} for key in cfg["domains"] for r in range(cfg["gpt_runs"])]
    res = pmap(_ident_proposals, jobs, cfg["gpt_workers"])
    write_jsonl(os.path.join(od, "proposal_runs.jsonl"), res)
    rate_rows = []
    for key in cfg["domains"]:
        rs = [r for r in res if r["domain"] == key and not r["error"]]
        for name in COMPETITORS[key]:
            rate_rows.append({"domain": key, "reference_hypothesis": name,
                              "is_intended": name in ("H_int", "S_inAisle"),
                              "proposed_in_runs": sum(r["proposed"].get(name, False) for r in rs), "n_runs": len(rs),
                              "n_errors": sum(1 for r in res if r["domain"] == key and r["error"])})
    write_csv(os.path.join(od, "A_proposal_rates.csv"), rate_rows)
    bayes = []
    if "cart" in cfg["domains"]:
        for r in res:
            if r["domain"] == "cart" and not r["error"]:
                bayes += _ident_bayes(r, cfg)
    write_csv(os.path.join(od, "B_posterior_given_gpt_sets.csv"), bayes)
    L = ["# GPT-5.5 identifiability", "", "## A. Proposal quality (initial from D1, refinement from D1–D3; "
         "equivalence = identical violation labels on the probe set)", "",
         "| Domain | Reference hypothesis | intended? | proposed in runs |", "|---|---|---|---|"]
    for r in rate_rows:
        L.append(f"| {r['domain']} | {r['reference_hypothesis']} | {'✓' if r['is_intended'] else ''} | "
                 f"{r['proposed_in_runs']}/{r['n_runs']} |")
    if bayes:
        L += ["", "## B. Bayesian identifiability given each run's GPT-5.5 hypothesis set (cart)", "",
              "| Sequence | t | mean P(intended) | runs with intended in set | mean accuracy |", "|---|---|---|---|---|"]
        for lab in ("control", "diagnostic", "reacquire"):
            for t in sorted({b["t"] for b in bayes}):
                bs = [b for b in bayes if b["sequence"] == lab and b["t"] == t]
                if bs:
                    L.append(f"| {lab} | {t} | {np.mean([b['P_intended'] for b in bs]):.3f} | "
                             f"{sum(b['intended_in_set'] for b in bs)}/{len(bs)} | {np.mean([b['accuracy'] for b in bs]):.2f} |")
    with open(os.path.join(od, "TABLE.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    cmds = {"estimate": cmd_estimate, "original": cmd_original, "threshold": cmd_threshold, "noise": cmd_noise,
            "identifiability": cmd_identifiability}
    if len(sys.argv) < 2 or sys.argv[1] not in cmds:
        sys.exit(f"usage: python -m experiments.gpt55 {{{','.join(cmds)}}} [--set key=value ...]")
    cmd = sys.argv.pop(1)
    DEFAULTS.update({"gpt_workers": 8, "gpt_noise_seeds": 10, "gpt_noise_methods": ["full", "fixed", "oracle", "mlci"]})
    cfg = parse_args(f"GPT-5.5 experiments: {cmd}")
    cfg["provider"] = "openai"
    t0 = time.time()
    cmds[cmd](cfg)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
