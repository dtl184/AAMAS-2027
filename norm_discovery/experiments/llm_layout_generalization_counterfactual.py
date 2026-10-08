"""Experiment 3 variant: counterfactual-environment guidance for GPT-5.5 norm proposal.

Identical to experiments/llm_layout_generalization.py except for ONE change: the generalisation paragraph inserted
into the c2 and c3 prompts is replaced by counterfactual-reasoning guidance (NEW_GUIDANCE below, verbatim as
specified).  Everything else -- layouts L1/L2/L3, training demonstrations, the 10 L3 diagnostics, probe sets, the
cached c1 prompts/responses, the paired initial hypothesis sets, the spurious coordinate competitors, priors,
likelihood, W grid, model and API-default generation settings -- is reused unchanged by importing the original
module and calling its own prepare/analyze functions with a different output directory and guidance string.

Fresh GPT-5.5 replicate/cache keys: genlayout_cf/<c2|c3>/<r>  (original experiment: genlayout/<c>/<r>).
Outputs: results/llm_layout_generalization_counterfactual/   (the original results directory is never written).

    cd /home/train/norm_discovery
    /home/train/anaconda3/bin/python -m experiments.llm_layout_generalization_counterfactual prepare   # no API
    bash -lc 'source ~/.bashrc && /home/train/anaconda3/bin/python -m experiments.llm_layout_generalization_counterfactual run'
    /home/train/anaconda3/bin/python -m experiments.llm_layout_generalization_counterfactual analyze   # no API
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time

import numpy as np

import experiments.llm_layout_generalization as BASE
from experiments.common import RESULTS, make_provider, parse_args, pmap, run_metadata, write_csv, write_json
from norms.hypotheses import Hypothesis
from norms.violations import count_violations

OLD_OUT = os.path.join(RESULTS, "llm_layout_generalization")
NEW_OUT = os.path.join(RESULTS, "llm_layout_generalization_counterfactual")
OLD_GUIDANCE = BASE.GUIDANCE
NEW_GUIDANCE = (
    "Generalization across hypothetical environment configurations. The demonstrations were collected in particular "
    "store environments, but the underlying social norms should remain applicable when the physical environment "
    "changes. Before proposing each candidate norm, consider hypothetical variations: the store may have different "
    "dimensions, shelves and displays may be rearranged or rotated, objects and service stations may be placed in "
    "different locations, entrances and exits may move, and task routes may change. A location that serves one "
    "function in the observed environment might serve a different function elsewhere.\n\n"
    "Distinguish genuine behavioral norms from regularities that depend only on observed coordinates, object "
    "locations, geometric configurations, or task routes. For each proposed rule, mentally test whether it remains "
    "normatively meaningful under these changes. Prefer behavioral and relational concepts that are invariant to "
    "incidental layout differences, but also avoid overly broad definitions that could prohibit legitimate behavior "
    "in other configurations. Revise or reject candidates that fail these counterfactual tests.\n\n"
    "These hypothetical environments are reasoning aids, not additional observations. Do not fabricate "
    "demonstrations from them. Every candidate must be consistent with the actual supplied demonstrations and "
    "executable using the available symbolic vocabulary. Include a brief justification explaining why each candidate "
    "should generalize.")
REPLICATE_PREFIX = "genlayout_cf"
FORBIDDEN_WORDS = ("aisle", "Aisle", "inAisle", "IN_AISLE")


def use_new():
    BASE.OUT = NEW_OUT
    BASE.GUIDANCE = NEW_GUIDANCE


# ----------------------------------------------------------------------------------------------- prompt checks
def verify_prompts():
    """The new prompts must equal the original experiment's prompts except for the guidance paragraph."""
    for w in FORBIDDEN_WORDS:
        assert w not in NEW_GUIDANCE, f"guidance contains {w!r}"
    l3 = BASE.spec_L3()
    for t in l3.heldout:                                   # no L3 trajectory names in the guidance
        assert t.name not in NEW_GUIDANCE
    with open(os.path.join(OLD_OUT, "prompts.jsonl")) as f:
        old = {r["run"]: r for r in map(json.loads, f)}
    with open(os.path.join(NEW_OUT, "prompts.jsonl")) as f:
        new = {r["run"]: r for r in map(json.loads, f)}
    l2 = BASE.spec_L2()
    l2_items = set(l2.domain.layout.items)
    report = []
    for r in sorted(new):
        o, n = old[r], new[r]
        assert n["c1"] == o["c1"], f"c1 prompt changed (run {r})"
        for c in ("c2", "c3"):
            assert OLD_GUIDANCE in o[c] and NEW_GUIDANCE in n[c]
            assert o[c].replace(OLD_GUIDANCE, "<G>") == n[c].replace(NEW_GUIDANCE, "<G>"), f"{c} differs beyond guidance (run {r})"
        c2 = n["c2"]
        n_demo = c2.count("Demonstration ")
        assert n_demo == 3 and "Store B" not in c2 and "store B" not in c2, f"c2 run {r} sees non-L1 demonstrations"
        assert not any(f"required items: {it}" in c2 for it in l2_items), "c2 contains L2 demonstrations"
        assert n["c3"].count("Store A, demonstration") == 3 and n["c3"].count("Store B, demonstration") == 3
        report.append({"run": r, "c1_identical_to_original": True, "c2_identical_except_guidance": True,
                       "c3_identical_except_guidance": True, "c2_demonstrations": n_demo, "c2_contains_L2_or_L3": False,
                       "c3_demonstrations": "3 x L1 + 3 x L2", "guidance_chars_old": len(OLD_GUIDANCE),
                       "guidance_chars_new": len(NEW_GUIDANCE)})
    write_json(os.path.join(NEW_OUT, "prompt_verification.json"), report)
    return report


# ----------------------------------------------------------------------------------------------- API calls
def _call(job):
    cfg, cond, r, prompt = job
    prov = make_provider(cfg, replicate=f"{REPLICATE_PREFIX}/{cond}/{r}",
                         run_info={"experiment": "llm_layout_generalization_counterfactual", "condition": cond,
                                   "run_index": r})
    try:
        out = prov._ask(prompt, "refinement", 1, 3 if cond == "c2" else 6)
        err = None
    except Exception as e:                                   # recorded; never replaced by another provider/model
        out, err = [], f"{type(e).__name__}: {e}"
    rec = {"condition": cond, "run_index": r, "error": err, "proposals": out, "transcript": prov.transcript,
           "llm_calls": [{k: v for k, v in c.items() if k != "output_text"} for c in prov.complete.calls]}
    d = os.path.join(NEW_OUT, "llm", cond, f"run_{r:02d}")
    os.makedirs(d, exist_ok=True)
    write_json(os.path.join(d, "transcript.json"), rec)
    return rec


def prepare(cfg):
    use_new()
    os.makedirs(NEW_OUT, exist_ok=True)
    out = BASE.prepare(cfg, verbose=False)
    rep = verify_prompts()
    with open(os.path.join(NEW_OUT, "setup.json")) as f:
        setup = json.load(f)
    setup["guidance_paragraph_previous_experiment"] = OLD_GUIDANCE
    setup["varies"] = {"relative to the previous experiment": "only the guidance paragraph of c2 and c3 (c1 unchanged)",
                       "c1 -> c2": "counterfactual guidance paragraph inserted",
                       "c2 -> c3": "L2 demonstrations B1-B3 added; demonstration block labelled store A / store B"}
    setup["replicate_keys"] = f"{REPLICATE_PREFIX}/<c2|c3>/<r>"
    write_json(os.path.join(NEW_OUT, "setup.json"), setup)
    print(f"prompt verification passed for {len(rep)} runs: c1 identical to the original experiment; c2/c3 identical "
          f"except the guidance paragraph; c2 contains only the 3 L1 demonstrations.")
    print(json.dumps({"planned_new_api_calls": 2 * BASE.N_RUNS, "c1": "0 (cached original responses)"}))
    return out


def run(cfg):
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY is not set (source ~/.bashrc); no fallback provider is used.")
    _, _, _, _, _, prompts = prepare(cfg)
    write_json(os.path.join(NEW_OUT, "metadata.json"), run_metadata("llm_layout_generalization_counterfactual", cfg))
    jobs = [(cfg, c, p["run"], p[c]) for p in prompts for c in ("c2", "c3")]
    res = pmap(_call, jobs, 8)
    calls = [c for r in res for c in r["llm_calls"]]
    fresh = [c for c in calls if not c.get("cache_hit")]
    usage = {"calls": len(calls), "fresh": len(fresh),
             "input_tokens": sum((c.get("usage") or {}).get("input_tokens", 0) for c in fresh),
             "output_tokens": sum((c.get("usage") or {}).get("output_tokens", 0) for c in fresh),
             "models": sorted({c.get("returned_model") for c in calls if c.get("returned_model")}),
             "parse_retries": sum(max(0, len(r["transcript"][-1].get("attempts", [])) - 1) for r in res if r["transcript"]),
             "errors": [(r["condition"], r["run_index"], r["error"][:300]) for r in res if r["error"]]}
    write_json(os.path.join(NEW_OUT, "api_usage.json"), usage)
    print(json.dumps(usage, indent=1))


# ----------------------------------------------------------------------------------------------- analysis
def _map_hypothesis(out_dir, row, c1_initial, spur1, spur3):
    """Recover the MAP hypothesis object of a run from its saved proposals / initial set / spurious set."""
    cands = [Hypothesis.from_dict(d) for d in c1_initial] + (spur3 if row["condition"] == "c3" else spur1)
    if row["condition"] == "c1":
        cands += [Hypothesis.from_dict(d) for d in BASE.cached_c1(row["run_index"])["refined"]]
    else:
        with open(os.path.join(out_dir, "llm", row["condition"], f"run_{row['run_index']:02d}", "transcript.json")) as f:
            cands += [Hypothesis.from_dict(d) for d in json.load(f)["proposals"] if "norm" in d]
    return next(h for h in cands if h.hid == row["map_id"])


def false_positive_analysis(out_dir):
    """FP / FN of each run's MAP hypothesis on the L1 held-out set and L3 diagnostics, attributed to the MAP kind."""
    l1, l2, l3 = BASE.setup()
    spur1 = BASE.spurious_hypotheses([(l1.domain, l1.training)])
    spur3 = BASE.spurious_hypotheses([(l1.domain, l1.training), (l2.domain, l2.training)])
    with open(os.path.join(out_dir, "runs.jsonl")) as f:
        rows = [json.loads(l) for l in f]
    out = []
    for row in rows:
        if row["status"] != "ok":
            continue
        h = _map_hypothesis(out_dir, row, BASE.cached_c1(row["run_index"])["initial"], spur1, spur3)
        rec = {"condition": row["condition"], "run_index": row["run_index"], "map_kind": row["map_kind"],
               "map_is_spurious": row["map_is_spurious"], "map_description": row["map_description"]}
        for tag, spec in (("L1", l1), ("L3", l3)):
            pred = [count_violations(spec.domain, h, t) > 0 for t in spec.heldout]
            truth = [bool(t.label) for t in spec.heldout]
            rec[f"{tag}_FP"] = sum(p and not y for p, y in zip(pred, truth))
            rec[f"{tag}_FN"] = sum(y and not p for p, y in zip(pred, truth))
            rec[f"{tag}_FP_trajectories"] = [t.name for t, p, y in zip(spec.heldout, pred, truth) if p and not y]
        rec["overbroad_relational_FP"] = (row["map_kind"] == "relational" and not row["selection_success"]
                                          and (rec["L1_FP"] + rec["L3_FP"]) > 0)
        out.append(rec)
    return out


def analyze(cfg):
    use_new()
    BASE.analyze(cfg)                        # the original pipeline, writing to NEW_OUT
    fp_new = false_positive_analysis(NEW_OUT)
    fp_old = false_positive_analysis(OLD_OUT)
    write_csv(os.path.join(NEW_OUT, "false_positives.csv"), fp_new)
    write_csv(os.path.join(NEW_OUT, "false_positives_previous_experiment.csv"), fp_old)
    comparison(fp_old, fp_new)


def _summary(out_dir):
    with open(os.path.join(out_dir, "summary.csv")) as f:
        return {r["condition"]: r for r in csv.DictReader(f)}


def _runs(out_dir):
    with open(os.path.join(out_dir, "runs.jsonl")) as f:
        return [json.loads(l) for l in f]


def comparison(fp_old, fp_new):
    so, sn = _summary(OLD_OUT), _summary(NEW_OUT)
    ro, rn = _runs(OLD_OUT), _runs(NEW_OUT)

    def stats(runs, fps, c):
        rs = [r for r in runs if r["condition"] == c and r["status"] == "ok"]
        fp = [x for x in fps if x["condition"] == c]
        n = len(rs)
        if not n:
            return None
        return {"n": n,
                "coord": np.mean([r["frac_coordinate_dependent"] for r in rs]),
                "rel": np.mean([r["frac_relational"] for r in rs]),
                "prop": sum(r["proposal_success"] for r in rs), "prop_L1": sum(r["proposal_success_L1_only"] for r in rs),
                "sel": sum(r["selection_success"] for r in rs), "spur": sum(r["map_is_spurious"] for r in rs),
                "P": np.mean([r["P_intended_equivalent"] for r in rs]),
                "L1": np.mean([r["L1_heldout_acc"] for r in rs]), "L1u": np.mean([r["L1_unseen_aisle_acc"] for r in rs]),
                "L3": np.mean([r["L3_acc"] for r in rs]), "L3f1": np.mean([r["L3_f1"] for r in rs]),
                "gen": sum(r["generalization_success"] for r in rs),
                "fp_runs": sum(x["overbroad_relational_FP"] for x in fp),
                "fp_L3": sum(x["L3_FP"] for x in fp), "fp_L1": sum(x["L1_FP"] for x in fp),
                "fp_L3_rel": sum(x["L3_FP"] for x in fp if x["map_kind"] == "relational"),
                "fn_L3": sum(x["L3_FN"] for x in fp)}

    rows = []
    for c in ("c1", "c2", "c3"):
        for tag, runs, fps in (("previous guidance", ro, fp_old), ("counterfactual guidance", rn, fp_new)):
            s = stats(runs, fps, c)
            if s is None:
                rows.append({"condition": c, "experiment": tag, "status": "not run"})
                continue
            rows.append({"condition": c, "experiment": tag, "n_runs": s["n"],
                         "frac_coordinate_proposals": round(s["coord"], 3), "frac_relational_proposals": round(s["rel"], 3),
                         "proposal_success": f"{s['prop']}/{s['n']}", "proposal_success_L1_only": f"{s['prop_L1']}/{s['n']}",
                         "selection_success": f"{s['sel']}/{s['n']}", "map_spurious": f"{s['spur']}/{s['n']}",
                         "mean_P_intended_equivalent": round(s["P"], 3), "L1_acc": round(s["L1"], 3),
                         "L1_unseen_acc": round(s["L1u"], 3), "L3_acc": round(s["L3"], 3), "L3_f1": round(s["L3f1"], 3),
                         "all_L3_correct": f"{s['gen']}/{s['n']}",
                         "runs_with_overbroad_relational_MAP_FP": f"{s['fp_runs']}/{s['n']}",
                         "L3_false_positives_total": s["fp_L3"], "L3_FP_from_relational_MAPs": s["fp_L3_rel"],
                         "L1_false_positives_total": s["fp_L1"], "L3_false_negatives_total": s["fn_L3"]})
    write_csv(os.path.join(NEW_OUT, "comparison_with_previous.csv"), rows)
    # LaTeX table
    name = {"c1": "Single layout, no guidance", "c2": "Single layout + guidance", "c3": "Two layouts + guidance"}
    L = [r"\begin{table}[t]", r"\centering\small",
         r"\caption{GPT-5.5 norm proposal across store layouts with the previous generalisation guidance (prev.) and "
         r"counterfactual-environment guidance (CF); 10 independent calls per condition. Proposal: a proposed hypothesis "
         r"is behaviourally equivalent to the intended norm on all three layouts. Selection: the Bayesian MAP is. "
         r"L3: test layout never shown to the LLM. c1 is identical in both experiments.}",
         r"\label{tab:layout-generalization-cf}", r"\begin{tabular}{llcccccc}", r"\toprule",
         r"Condition & Guid. & Coord. & Proposal & Selection & Spurious MAP & L1 acc. & L3 acc. / F1 \\", r"\midrule"]
    for r in rows:
        if r.get("status") == "not run":
            continue
        if r["condition"] == "c1" and r["experiment"] == "counterfactual guidance":
            continue
        g = "--" if r["condition"] == "c1" else ("prev." if r["experiment"] == "previous guidance" else "CF")
        L.append(f"{name[r['condition']]} & {g} & {r['frac_coordinate_proposals']:.2f} & {r['proposal_success']} & "
                 f"{r['selection_success']} & {r['map_spurious']} & {r['L1_acc']:.2f} & {r['L3_acc']:.2f} / {r['L3_f1']:.2f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(os.path.join(NEW_OUT, "table.tex"), "w") as f:
        f.write("\n".join(L) + "\n")
    for r in rows:
        print(json.dumps(r))


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("prepare", "run", "analyze"):
        sys.exit("usage: python -m experiments.llm_layout_generalization_counterfactual {prepare,run,analyze}")
    st = sys.argv.pop(1)
    cfg = parse_args("Counterfactual-guidance layout generalisation")
    cfg["provider"] = "openai"
    t0 = time.time()
    {"prepare": prepare, "run": run, "analyze": analyze}[st](cfg)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
