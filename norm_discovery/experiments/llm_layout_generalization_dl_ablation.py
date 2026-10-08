"""Description-length prior ablation for Experiment 3 (layout-independent norm proposal).

The ONLY varied factor is the description-length coding used in the prior
    P(h | H) ∝ exp(-lambda L(alpha_h) - gamma L(N_h)),  lambda = gamma = 0.2, normalised over the candidate set:
  A "nodes"  (original): every atom 1 (incl. at(x,y)), every operator 1, numeric comparison 3
  B "tokens" (existing implementation abstractions/base.py::formula_size): as A but at(x,y) / cartAt(x,y) cost 3
Candidate sets are rebuilt exactly as in experiments/llm_layout_generalization.py::analyze (initial GPT-5.5 set ∪ valid
refined GPT-5.5 proposals ∪ the two spurious coordinate competitors, in that order), from the SAVED proposals of
  * results/llm_layout_generalization/                 (original guidance; c1 = cached original responses)
  * results/llm_layout_generalization_counterfactual/  (counterfactual guidance; c1 identical, evaluated once)
Same demonstrations, max-ent likelihood, W grid, probe sets and evaluation sets.  No OpenAI calls: the key is removed
from this process and no provider is constructed.

    cd /home/train/norm_discovery && /home/train/anaconda3/bin/python -m experiments.llm_layout_generalization_dl_ablation
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import time

os.environ.pop("OPENAI_API_KEY", None)

import numpy as np
from scipy.special import logsumexp

import experiments.llm_layout_generalization as BASE
from experiments.common import RESULTS, parse_args, write_csv, write_json, write_jsonl
from inference.likelihood import ZComputer
from norms.hypotheses import Hypothesis
from norms.violations import count_violations

OUT = os.path.join(RESULTS, "llm_layout_generalization_dl_ablation")
EXPS = {"original": os.path.join(RESULTS, "llm_layout_generalization"),
        "counterfactual": os.path.join(RESULTS, "llm_layout_generalization_counterfactual")}
CODINGS = ("nodes", "tokens")


def sha_files(pattern):
    h = hashlib.sha256()
    files = sorted(glob.glob(pattern, recursive=True))
    for f in files:
        with open(f, "rb") as fh:
            h.update(f.encode() + fh.read())
    return h.hexdigest(), len(files)


def refined_set(exp_dir, cond, r, specs):
    """Exactly the validation of BASE.analyze: from_dict, vocabulary check, executable in every layout."""
    if cond == "c1":
        dicts = BASE.cached_c1(r)["refined"]
    else:
        with open(os.path.join(exp_dir, "llm", cond, f"run_{r:02d}", "transcript.json")) as f:
            rec = json.load(f)
        if rec["error"]:
            return None
        dicts = rec["proposals"]
    out = []
    for d in dicts:
        try:
            h = Hypothesis.from_dict(d)
            h.validate(BASE.VOCAB[1])
            for s in specs.values():
                for t in (s.training or s.heldout)[:1]:
                    count_violations(s.domain, h, t)
            out.append(h)
        except Exception:
            pass
    return out


def log_prior(hyps, coding, lam, gamma):
    La = np.array([h.L_alpha(coding) for h in hyps], float)
    Ln = np.array([h.L_norm(coding) for h in hyps], float)
    lp = -lam * La - gamma * Ln
    return lp - logsumexp(lp), La, Ln


def loglik_matrix(hyps, data, zcs):
    """sum_t log P(tau_t | h, w) for every (h, w): independent of the prior coding."""
    return np.sum([np.array([zcs[name].loglik(h, t) for h in hyps]) for name, t in data], axis=0)


def evaluate_map(h, l1, l3, th, specs):
    m1, _ = BASE.eval_on(l1, h, l1.heldout)
    unseen = [t for t in l1.heldout if t.name in l1.unseen_names]
    mu, _ = BASE.eval_on(l1, h, unseen)
    m3, pred3 = BASE.eval_on(l3, h, l3.heldout)
    return {"L1_acc": m1["accuracy"], "L1_f1": m1["violation_f1"], "L1_unseen_acc": mu["accuracy"],
            "L3_acc": m3["accuracy"], "L3_f1": m3["violation_f1"], "L3_all_correct": m3["accuracy"] == 1.0,
            "L3_FP": m3["fp"], "L3_FN": m3["fn"], "L1_FP": m1["fp"]}


def main():
    cfg = parse_args("Description-length prior ablation for Experiment 3")
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    lam, gamma = cfg["lam"], cfg["gamma"]
    cache_before = len(os.listdir(os.path.join(RESULTS, "llm_cache")))
    hashes_before = {k: sha_files(os.path.join(d, "llm", "**", "transcript.json")) for k, d in EXPS.items()}
    probe_before = sha_files(os.path.join(RESULTS, "probe_sets", "*.json"))
    l1, l2, l3 = BASE.setup()
    specs = {"L1": l1, "L2": l2, "L3": l3}
    spur1 = BASE.spurious_hypotheses([(l1.domain, l1.training)])
    spur3 = BASE.spurious_hypotheses([(l1.domain, l1.training), (l2.domain, l2.training)])
    th = BASE.intended()
    zcs = {k: ZComputer(s.domain, cfg["beta"], np.round(np.arange(0, cfg["w_max"] + 1e-9, cfg["w_step"]), 6),
                        cache_dir=os.path.join(RESULTS, "zcache")) for k, s in specs.items()}
    W = zcs["L1"].W
    rows, diag, checks = [], [], {"reproduction": [], "likelihood_identical": True}
    jobs = [("original", "c1"), ("original", "c2"), ("original", "c3"), ("counterfactual", "c2"), ("counterfactual", "c3")]
    for exp, cond in jobs:
        with open(os.path.join(EXPS[exp], "runs.jsonl")) as f:
            saved = {(r["condition"], r["run_index"]): r for r in map(json.loads, f)}
        for r in range(BASE.N_RUNS):
            refined = refined_set(EXPS[exp], cond, r, specs)
            if refined is None:
                rows.append({"guidance": exp, "condition": cond, "run_index": r, "status": "error in saved run"})
                continue
            initial = [Hypothesis.from_dict(d) for d in BASE.cached_c1(r)["initial"]]
            spur = spur3 if cond == "c3" else spur1
            cand = initial + refined + spur
            assert {h.hid for h in spur} <= {h.hid for h in cand}
            data = [("L1", t) for t in l1.training] + ([("L2", t) for t in l2.training] if cond == "c3" else [])
            LL = loglik_matrix(cand, data, zcs)
            LL2 = loglik_matrix(cand, data, zcs)                       # recomputed for the second coding
            checks["likelihood_identical"] &= bool(np.array_equal(LL, LL2))
            logml = logsumexp(LL, axis=1) - np.log(len(W))           # log P(D | h), uniform w prior
            eq = [all(s.equivalent(h, th) for s in specs.values()) for h in cand]
            proposed = any(eq[len(initial):len(initial) + len(refined)])
            per = {}
            for coding, LLc in (("nodes", LL), ("tokens", LL2)):
                lp, La, Ln = log_prior(cand, coding, lam, gamma)
                lj = lp[:, None] - np.log(len(W)) + LLc
                lj -= logsumexp(lj)
                post = np.exp(logsumexp(lj, axis=1))
                i = int(np.argmax(post))
                hm = cand[i]
                ev = evaluate_map(hm, l1, l3, th, specs)
                per[coding] = {"post": post, "lp": lp, "imap": i}
                rows.append({"guidance": exp, "condition": cond, "run_index": r, "status": "ok", "coding": coding,
                             "intended_proposed": proposed, "map_id": hm.hid, "map_description": hm.describe(),
                             "map_kind": BASE.classify_atoms(hm), "map_is_intended": eq[i],
                             "map_is_spurious": hm.hid.startswith("SPUR"), "map_L_alpha": int(La[i]),
                             "map_L_norm": int(Ln[i]), "map_penalty": float(lam * La[i] + gamma * Ln[i]),
                             "map_posterior": float(post[i]),
                             "P_intended_equivalent": float(sum(p for p, e in zip(post, eq) if e)), **ev,
                             "n_candidates": len(cand)})
                if coding == "nodes":                                  # must reproduce the saved experiment exactly
                    s = saved[(cond, r)]
                    ok = (s["map_id"] == hm.hid and s["selection_success"] == eq[i]
                          and abs(s["L3_acc"] - ev["L3_acc"]) < 1e-12 and abs(s["L1_heldout_acc"] - ev["L1_acc"]) < 1e-12
                          and abs(s["P_intended_equivalent"] - rows[-1]["P_intended_equivalent"]) < 1e-9)
                    checks["reproduction"].append({"guidance": exp, "condition": cond, "run_index": r, "ok": bool(ok)})
            # ---- odds decomposition: best intended-equivalent vs the original (node-coding) MAP competitor
            if proposed:
                j_coord = per["nodes"]["imap"]
                for coding in CODINGS:
                    post, lp = per[coding]["post"], per[coding]["lp"]
                    eq_idx = [k for k, e in enumerate(eq) if e]
                    j_cor = max(eq_idx, key=lambda k: post[k])
                    if j_cor == j_coord:
                        continue
                    d_ll = float(logml[j_cor] - logml[j_coord])
                    d_lp = float(lp[j_cor] - lp[j_coord])
                    diag.append({"guidance": exp, "condition": cond, "run_index": r, "coding": coding,
                                 "correct_id": cand[j_cor].hid, "correct_description": cand[j_cor].describe(),
                                 "correct_L_alpha": cand[j_cor].L_alpha(coding), "correct_L_norm": cand[j_cor].L_norm(coding),
                                 "competitor_id": cand[j_coord].hid, "competitor_description": cand[j_coord].describe(),
                                 "competitor_L_alpha": cand[j_coord].L_alpha(coding),
                                 "competitor_L_norm": cand[j_coord].L_norm(coding),
                                 "log_likelihood_diff": d_ll, "log_prior_diff": d_lp, "log_posterior_odds": d_ll + d_lp,
                                 "correct_wins_pairwise": d_ll + d_lp > 0,
                                 "map_under_coding": cand[per[coding]["imap"]].hid})
    # c1 is identical in both experiments: reuse the 'original' c1 rows for the counterfactual block
    rows += [dict(r, guidance="counterfactual") for r in rows if r["guidance"] == "original" and r["condition"] == "c1"]
    write_jsonl(os.path.join(OUT, "runs.jsonl"), rows)
    write_csv(os.path.join(OUT, "runs.csv"), rows)
    write_csv(os.path.join(OUT, "odds_decomposition.csv"), diag)
    # ---------------------------------------------------------------- validation
    checks["reproduction_all_ok"] = all(c["ok"] for c in checks["reproduction"])
    checks["n_reproduced"] = len(checks["reproduction"])
    checks["saved_proposals_unchanged"] = {k: hashes_before[k] == sha_files(os.path.join(d, "llm", "**", "transcript.json"))
                                           for k, d in EXPS.items()}
    checks["no_new_api_calls"] = {"OPENAI_API_KEY_in_process": bool(os.environ.get("OPENAI_API_KEY")),
                                  "llm_cache_files_before": cache_before,
                                  "llm_cache_files_after": len(os.listdir(os.path.join(RESULTS, "llm_cache")))}
    checks["spurious_retained"] = [h.describe() for h in spur1] + [h.describe() for h in spur3]
    with open(os.path.join(EXPS["original"], "setup.json")) as f:
        setup = json.load(f)
    checks["L3_diagnostics_unchanged"] = [x["actions"] for x in setup["L3_diagnostics"]] == \
        [[l3.domain.action_name(a) for a in t.actions] for t in l3.heldout]
    checks["probe_sets_unchanged"] = probe_before == sha_files(os.path.join(RESULTS, "probe_sets", "*.json"))
    # description-length spot checks (computed by hand from the definitions)
    by = {h.hid: h for h in spur1}
    hand = {"SPUR_memorised_cells (4 cells)": ((0, 8), (0, 16), by["SPUR_memorised_cells"]),
            "SPUR_aisle_region_coords (9 cells)": ((0, 13), (0, 31), by["SPUR_aisle_region_coords"]),
            "intended S_inAisle": ((7, 4), (7, 4), th)}
    checks["description_length_spot_checks"] = {
        k: {"expected_nodes": [exp[0] if i == 0 else exp[1] for i, exp in [(0, n), (1, n)]],
            "computed_nodes": [h.L_alpha("nodes"), h.L_norm("nodes")],
            "expected_tokens": [t[0] if i == 0 else t[1] for i, t in [(0, tk), (1, tk)]],
            "computed_tokens": [h.L_alpha("tokens"), h.L_norm("tokens")]}
        for k, (n, tk, h) in hand.items()}
    checks["description_length_spot_checks_ok"] = all(
        v["expected_nodes"] == v["computed_nodes"] and v["expected_tokens"] == v["computed_tokens"]
        for v in checks["description_length_spot_checks"].values())
    write_json(os.path.join(OUT, "validation.json"), checks)
    summarise(rows, diag)
    print(json.dumps({k: v for k, v in checks.items() if k != "reproduction"}, indent=1, default=str))
    print(f"done in {time.time() - t0:.0f}s")


def paired_boot(d, n=10000, seed=0):
    d = np.asarray(d, float)
    rng = np.random.default_rng(seed)
    bs = rng.choice(d, size=(n, len(d)), replace=True).mean(1)
    return float(d.mean()), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))


def summarise(rows, diag):
    ok = [r for r in rows if r["status"] == "ok"]
    summ, paired = [], []
    for exp in ("original", "counterfactual"):
        for cond in ("c1", "c2", "c3"):
            by = {c: sorted([r for r in ok if r["guidance"] == exp and r["condition"] == cond and r["coding"] == c],
                            key=lambda r: r["run_index"]) for c in CODINGS}
            for c in CODINGS:
                rs = by[c]
                n = len(rs)
                summ.append({"guidance": exp, "condition": cond, "prior": c, "n_runs": n,
                             "correct_proposals": f"{sum(r['intended_proposed'] for r in rs)}/{n}",
                             "correct_map": f"{sum(r['map_is_intended'] for r in rs)}/{n}",
                             "spurious_map": f"{sum(r['map_is_spurious'] for r in rs)}/{n}",
                             "P_intended_mean": float(np.mean([r["P_intended_equivalent"] for r in rs])),
                             "L1_acc": float(np.mean([r["L1_acc"] for r in rs])),
                             "L1_unseen_acc": float(np.mean([r["L1_unseen_acc"] for r in rs])),
                             "L3_acc": float(np.mean([r["L3_acc"] for r in rs])),
                             "L3_f1": float(np.mean([r["L3_f1"] for r in rs])),
                             "L3_all_correct": f"{sum(r['L3_all_correct'] for r in rs)}/{n}",
                             "L3_false_positives": int(sum(r["L3_FP"] for r in rs)),
                             "map_kinds": dict(__import__("collections").Counter(r["map_kind"] for r in rs))})
            a, b = by["nodes"], by["tokens"]
            for m in ("P_intended_equivalent", "L3_acc", "L1_acc", "L1_unseen_acc", "L3_f1"):
                mu, lo, hi = paired_boot([y[m] - x[m] for x, y in zip(a, b)])
                paired.append({"guidance": exp, "condition": cond, "metric": m, "tokens_minus_nodes": mu,
                               "ci_lo": lo, "ci_hi": hi})
            paired.append({"guidance": exp, "condition": cond, "metric": "correct MAP (nodes->tokens)",
                           "gained": sum((not x["map_is_intended"]) and y["map_is_intended"] for x, y in zip(a, b)),
                           "lost": sum(x["map_is_intended"] and not y["map_is_intended"] for x, y in zip(a, b))})
    write_csv(os.path.join(OUT, "summary.csv"), summ)
    write_csv(os.path.join(OUT, "paired_differences.csv"), paired)
    # ---- per-run C2 table
    c2 = []
    for exp in ("original", "counterfactual"):
        for r in range(BASE.N_RUNS):
            x = next((y for y in ok if y["guidance"] == exp and y["condition"] == "c2" and y["run_index"] == r
                      and y["coding"] == "nodes"), None)
            y = next((y for y in ok if y["guidance"] == exp and y["condition"] == "c2" and y["run_index"] == r
                      and y["coding"] == "tokens"), None)
            if x is None or y is None:
                continue
            dn = next((d for d in diag if d["guidance"] == exp and d["condition"] == "c2" and d["run_index"] == r
                       and d["coding"] == "nodes"), None)
            dt = next((d for d in diag if d["guidance"] == exp and d["condition"] == "c2" and d["run_index"] == r
                       and d["coding"] == "tokens"), None)
            c2.append({"guidance": exp, "run": r, "intended_proposed": x["intended_proposed"],
                       "nodes_MAP": x["map_id"], "tokens_MAP": y["map_id"], "tokens_MAP_kind": y["map_kind"],
                       "nodes_P_intended": round(x["P_intended_equivalent"], 4),
                       "tokens_P_intended": round(y["P_intended_equivalent"], 4),
                       "nodes_log_prior_diff": None if dn is None else round(dn["log_prior_diff"], 3),
                       "tokens_log_prior_diff": None if dt is None else round(dt["log_prior_diff"], 3),
                       "log_likelihood_diff": None if dn is None else round(dn["log_likelihood_diff"], 3),
                       "nodes_log_odds": None if dn is None else round(dn["log_posterior_odds"], 3),
                       "tokens_log_odds": None if dt is None else round(dt["log_posterior_odds"], 3),
                       "L3_acc_nodes": x["L3_acc"], "L3_acc_tokens": y["L3_acc"], "L3_improved": y["L3_acc"] > x["L3_acc"],
                       "L3_FP_tokens": y["L3_FP"]})
    write_csv(os.path.join(OUT, "c2_per_run.csv"), c2)
    # ---- LaTeX
    name = {"c1": "C1 single layout, no guidance", "c2": "C2 single layout + guidance", "c3": "C3 two layouts + guidance"}
    L = [r"\begin{table}[t]", r"\centering\small",
         r"\caption{Description-length prior ablation for the layout experiment. Same saved GPT-5.5 proposals, candidate "
         r"sets, demonstrations and likelihood; only the coding of $L(\cdot)$ differs (nodes: \texttt{at(x,y)} = 1 unit; "
         r"tokens: \texttt{at(x,y)} = 3 units, i.e.\ integer constants counted consistently). 10 runs per row.}",
         r"\label{tab:dl-ablation}", r"\begin{tabular}{llcccccc}", r"\toprule",
         r"Guidance / condition & Prior & Correct prop. & Correct MAP & Spurious MAP & L1 acc. & L3 acc. & L3 F1 \\",
         r"\midrule"]
    for exp in ("original", "counterfactual"):
        for cond in ("c1", "c2", "c3"):
            if exp == "counterfactual" and cond == "c1":
                continue
            for c in CODINGS:
                s = next(x for x in summ if x["guidance"] == exp and x["condition"] == cond and x["prior"] == c)
                g = "" if cond == "c1" else (" (prev.)" if exp == "original" else " (CF)")
                L.append(f"{name[cond]}{g} & {c} & {s['correct_proposals']} & {s['correct_map']} & {s['spurious_map']} & "
                         f"{s['L1_acc']:.2f} & {s['L3_acc']:.2f} & {s['L3_f1']:.2f} \\\\")
        L.append(r"\midrule")
    L[-1] = r"\bottomrule"
    L += [r"\end{tabular}", r"\end{table}"]
    with open(os.path.join(OUT, "table.tex"), "w") as f:
        f.write("\n".join(L) + "\n")
    for s in summ:
        print(json.dumps(s))
    for p in paired:
        print(json.dumps(p))
    for c in c2:
        print(json.dumps(c))


if __name__ == "__main__":
    main()
