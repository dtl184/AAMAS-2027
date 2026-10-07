"""Phase 1/2: reconstruct the original experiments, print the reconstruction report, run sanity checks.

python -m experiments.reconstruct_original [--quick] [--set key=value ...]
"""
from __future__ import annotations

import itertools
import json
import os
import time

import numpy as np

from analysis import plotting as P
from environment.domains import get_spec
from environment.trajectories import is_valid
from experiments.common import (FIGURES, bootstrap_ci, flat, learner_kwargs, make_zc, out_dir, parse_args, pmap,
                                run_metadata, run_method, seed_for, write_csv, write_json, write_jsonl)
from inference.posterior import Posterior
from inference.predictive import evaluate_hypothesis
from norms.hypotheses import Hypothesis
from norms.violations import count_violations
from proposals import templates as T
from proposals.base import render_label
from proposals.deterministic import DeterministicProposalProvider

METHODS = ["fixed", "full", "oracle", "mlci"]


# ------------------------------------------------------------------------------------------ figures
def make_env_figures():
    P.setup()
    os.makedirs(FIGURES, exist_ok=True)
    paths = []
    for key, title in (("cart", "Exp. 1 store (cart-use norm)"), ("aisle", "Exp. 2 store (aisle-use norm)")):
        spec = get_spec(key)
        fig, axes = P.plt.subplots(1, 2, figsize=(10, 3.4 if key == "cart" else 4.0))
        P.draw_layout(axes[0], spec.domain.layout, title + ": layout")
        P.draw_layout(axes[1], spec.domain.layout, "training demonstrations")
        for i, t in enumerate(spec.training):
            P.draw_path(axes[1], spec.domain, t, color=P.SLOTS[i], offset=(i - 1) * 0.12, label=t.name)
        axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.04), ncol=3, fontsize=7)
        p = os.path.join(FIGURES, f"gridworld_{key}.png")
        P.save(fig, p)
        paths.append(p)
    return paths


# ------------------------------------------------------------------------------------------ report pieces
def hypothesis_rows(spec, cfg):
    """All candidate hypotheses (level 0 from tau_1, level 1 from D) with description lengths and priors."""
    prov = DeterministicProposalProvider()
    dicts = prov.propose_initial(spec, spec.training[:1]) + prov.refine(spec, spec.training, [], 1)
    hyps = [Hypothesis.from_dict(d) for d in dicts]
    for h in hyps:
        h.validate(spec.domain.vocab_levels[h.level])
    post = Posterior(make_zc(spec, cfg), cfg["lam"], cfg["gamma"], cfg["prior_mode"], cfg["dl_coding"])
    post.add_hypotheses(hyps)
    lp = post.log_prior_h()
    rows = []
    for h, l in zip(post.hyps, lp):
        ev = evaluate_hypothesis(spec, h)
        rows.append({"id": h.hid, "level": h.level, "intended": h.key == spec.true_hypothesis.key,
                     "norm": h.norm.describe(), "definitions": h.abstraction.describe(),
                     "L_alpha": h.L_alpha(), "L_N": h.L_norm(), "L_alpha_tokens": h.L_alpha("tokens"),
                     "L_N_tokens": h.L_norm("tokens"), "prior_joint": float(np.exp(l)),
                     "violations_on_training": [count_violations(spec.domain, h, t) for t in spec.training],
                     "heldout_accuracy_if_selected": ev["accuracy"], "heldout_f1_if_selected": ev["violation_f1"],
                     "unseen_accuracy_if_selected": ev["unseen_accuracy"], "json": h.to_dict()})
    return rows


def alpha0_expressivity(spec):
    """Exhaustive check of what norms over alpha_0 can achieve (verifies abstraction is needed)."""
    d, L = spec.domain, spec.domain.layout
    cells = L.free_cells
    cands = []
    at = T.at
    if spec.key == "cart":
        trig = ["START", "INTERACT"] + [{"and": ["INTERACT", at(c)]} for c in cells] + [at(c) for c in cells]
        dis = [{"and": ["INTERACT", at(c)]} for c in cells] + [at(c) for c in cells]
        for tr in trig:
            for di in dis:
                cands.append({"type": "obligation", "trigger": tr, "discharge": di})
        for c in cells:
            cands.append({"type": "prohibition", "condition": {"and": ["INTERACT", at(c)]}})
            cands.append({"type": "prohibition", "condition": at(c)})
    else:
        for c in cells:
            cands.append({"type": "prohibition", "condition": {"and": ["hasCart", at(c)]}})
            cands.append({"type": "prohibition", "condition": {"and": ["hasCart", "INTERACT", at(c)]}})
    seen_cells = {d.cell(s[0]) for t in spec.training for s in t.states(d)}
    best, best_unseen_supported, n_consistent = None, None, 0
    reacq_ok = 0
    for i, n in enumerate(cands):
        h = Hypothesis.from_dict({"id": f"a0_{i}", "abstraction": {}, "norm": n})
        if any(count_violations(d, h, t) for t in spec.training):
            continue
        n_consistent += 1
        ev = evaluate_hypothesis(spec, h)
        if best is None or ev["accuracy"] > best[1]["accuracy"]:
            best = (h.norm.describe(), ev)
        if spec.key == "cart":
            preds = ev["predictions"]
            if all(preds[t.name] == t.label for t in spec.heldout if t.meta.get("kind") in ("reacquire", "compliant")):
                reacq_ok += 1
        else:
            named = {tuple(map(int, a[3:-1].split(","))) for a in h.primitive_atoms() if a.startswith("at(")}
            if named <= seen_cells:          # rule only names cells that appear in some training trajectory
                if best_unseen_supported is None or ev["unseen_accuracy"] > best_unseen_supported[1]["unseen_accuracy"]:
                    best_unseen_supported = (h.norm.describe(), ev)
    out = {"n_candidate_alpha0_norms": len(cands), "n_consistent_with_training": n_consistent,
           "best_heldout_accuracy_any_consistent_alpha0_norm (oracle-selected on held-out!)":
               {"norm": best[0], "accuracy": best[1]["accuracy"], "violation_f1": best[1]["violation_f1"],
                "unseen_accuracy": best[1]["unseen_accuracy"]}}
    if spec.key == "cart":
        out["n_consistent_alpha0_norms_correct_on_all_compliant_and_reacquire_cases"] = reacq_ok
    else:
        out["best_unseen_accuracy_among_consistent_rules_naming_only_cells_seen_in_training"] = {
            "norm": best_unseen_supported[0], "unseen_accuracy": best_unseen_supported[1]["unseen_accuracy"],
            "accuracy": best_unseen_supported[1]["accuracy"]}
    return out


def z_check(spec, cfg):
    """Verify the fixed-point partition function against a direct sparse solve."""
    import scipy.sparse as sp
    import scipy.sparse.linalg as spl
    from inference.likelihood import ProductSystem
    h = spec.true_hypothesis
    task = spec.training[0].task
    zc = make_zc(spec, cfg)
    lz = zc.logZ(h, task)
    ps = ProductSystem(spec.domain, task, h, cfg["beta"])
    out = []
    for j in (0, 10, 40):
        w = zc.W[j]
        A = sp.identity(ps.n, format="csc") - (ps.M0 + np.exp(-cfg["beta"] * w) * ps.M1)
        z = spl.spsolve(A, ps.b0 + np.exp(-cfg["beta"] * w) * ps.b1)
        out.append({"w": float(w), "logZ_fixed_point": float(lz[j]), "logZ_direct": float(np.log(z[ps.start]))})
    return out


# ------------------------------------------------------------------------------------------ main
def _run_job(job):
    key, method, seed, cfg = job
    spec = get_spec(key)
    if seed is None:
        demos = spec.training
    else:   # same tasks as the original demonstrations, optimal paths with random tie-breaking
        rng = np.random.default_rng(seed)
        demos = [spec.clean_demo(t.task, rng, name=t.name) for t in spec.training]
    r = run_method(spec, method, demos, cfg, record_prefix_eval=(seed is None))
    r.update({"domain": key, "seed": seed, "demos": [t.to_dict(spec.domain) for t in demos] if seed is not None else None})
    return r


def main():
    cfg = parse_args("Reconstruct the original experiments and run sanity checks")
    od = out_dir(cfg, "reconstruction")
    t0 = time.time()
    write_json(os.path.join(od, "metadata.json"), run_metadata("reconstruct_original", cfg))
    figs = make_env_figures()
    report = {"figures": figs, "domains": {}}
    for key in cfg["domains"]:
        spec = get_spec(key)
        d = spec.domain
        report["domains"][key] = {
            "layout_ascii": d.layout.ascii().split("\n"),
            "layout_legend": "'#' shelf, 'i' shelf holding an item, E entrance/start, X exit, C cart station, R cart return",
            "items": {n: list(c) for n, c in d.layout.items.items()},
            "ground_truth_aisles(evaluation only)": {k: v for k, v in d.layout.aisles.items()},
            "mechanics": {"exit_requires_cart": d.exit_requires_cart, "cart_reach": d.cart_reach,
                          "hand_capacity": d.hand_capacity, "action_cost": 1},
            "vocabulary_levels": {str(k): list(v) for k, v in d.vocab_levels.items()},
            "hidden_norm": spec.true_hypothesis.describe(),
            "training": [{**t.to_dict(d), "compact": t.pretty(d),
                          "alpha0_view": render_label(d, t, d.vocab_levels[0])} for t in spec.training],
            "heldout": [{**t.to_dict(d), "compact": t.pretty(d), "unseen_structure": t.name in spec.unseen_names}
                        for t in spec.heldout],
            "candidate_hypotheses": hypothesis_rows(spec, cfg),
            "alpha0_expressivity": alpha0_expressivity(spec),
            "partition_function_check": z_check(spec, cfg),
            "validity": all(is_valid(d, t)[0] for t in spec.training + spec.heldout),
        }
    # ---- original protocol (deterministic) + randomized tie-breaking supplement
    jobs = [(k, m, None, cfg) for k in cfg["domains"] for m in METHODS]
    n_rand = 20 if not cfg["_args"].get("quick") else 2
    jobs += [(k, m, seed_for(cfg, "orig_random", k, s), cfg) for k in cfg["domains"] for m in METHODS
             for s in range(n_rand)]
    res = pmap(_run_job, jobs, cfg["workers"])
    write_jsonl(os.path.join(od, "original_runs.jsonl"), [r for r in res if r["seed"] is None])
    write_jsonl(os.path.join(od, "randomized_original_runs.jsonl"), [r for r in res if r["seed"] is not None])
    table = [{"domain": r["domain"], **flat(r)} for r in res if r["seed"] is None]
    write_csv(os.path.join(od, "original_results.csv"), table)
    agg = []
    for k in cfg["domains"]:
        for m in METHODS:
            rs = [r for r in res if r["seed"] is not None and r["domain"] == k and r["method"] == m]
            row = {"domain": k, "method": m, "n_seeds": len(rs)}
            for metric in ("accuracy", "violation_f1", "unseen_accuracy"):
                vals = [r[metric] for r in rs if r.get(metric) is not None]
                if vals:
                    mu, lo, hi = bootstrap_ci(vals, cfg["bootstrap"])
                    row.update({f"{metric}_mean": mu, f"{metric}_sd": float(np.std(vals)), f"{metric}_ci_lo": lo,
                                f"{metric}_ci_hi": hi})
            if m != "mlci":
                row["rate_map_intended"] = float(np.mean([bool(r["map_is_intended"]) for r in rs]))
                row["rate_refined"] = float(np.mean([(r["n_refinements"] or 0) > 0 for r in rs]))
                row["mean_P_intended"] = float(np.mean([r["P_intended"] or 0.0 for r in rs]))
            agg.append(row)
    write_csv(os.path.join(od, "randomized_original_summary.csv"), agg)
    # ---- MLCI epsilon sensitivity (supplementary; NOT used to choose epsilon)
    sens = []
    from baselines.mlci import MLCI, MLCIConfig
    for k in cfg["domains"]:
        spec = get_spec(k)
        for eps in (0.1, 1.0, "log_candidates", 15.0):
            ev = MLCI(spec, MLCIConfig(beta=cfg["beta"], epsilon=eps)).fit(spec.training).evaluate()
            sens.append({"domain": k, "epsilon": eps, "epsilon_used": ev["epsilon"], "accuracy": ev["accuracy"],
                         "violation_f1": ev["violation_f1"], "unseen_accuracy": ev["unseen_accuracy"],
                         "n_constraints": ev["n_constraints"], "constraints": ev["constraints"]})
    write_csv(os.path.join(od, "mlci_epsilon_sensitivity.csv"), sens)
    # ---- sanity checks
    checks = {}
    for k in cfg["domains"]:
        spec = get_spec(k)
        orig = {r["method"]: r for r in res if r["seed"] is None and r["domain"] == k}
        hyps = {h["id"]: h for h in report["domains"][k]["candidate_hypotheses"]}
        intended_row = next(h for h in hyps.values() if h["intended"])
        c = {"intended_hypothesis_heldout_accuracy": intended_row["heldout_accuracy_if_selected"],
             "intended_hypothesis_consistent_with_training": not any(intended_row["violations_on_training"]),
             "oracle_map": orig["oracle"]["map_id"], "oracle_map_is_intended": orig["oracle"]["map_is_intended"],
             "oracle_P_intended": orig["oracle"]["P_intended"], "oracle_accuracy": orig["oracle"]["accuracy"],
             "fixed_accuracy": orig["fixed"]["accuracy"], "fixed_map": orig["fixed"]["map_id"],
             "full_refined": orig["full"]["n_refinements"] > 0, "full_trace": [
                 {kk: s[kk] for kk in ("t", "log_p", "log_b", "log_ratio_p_over_b", "triggered")}
                 for s in orig["full"]["trace"]]}
        if k == "aisle":
            mem = Hypothesis.from_dict(next(h for h in T.exp2_level1([(3, 1), (3, 2), (6, 1), (6, 2)])
                                            if h["id"] == "S_memorised"))
            c["unseen_test"] = {"memorised_A1_A2_cells_unseen_accuracy": evaluate_hypothesis(spec, mem)["unseen_accuracy"],
                                "memorised_A1_A2_cells_seen_accuracy": _seen_acc(spec, mem),
                                "structural_inAisle_unseen_accuracy": evaluate_hypothesis(spec, spec.true_hypothesis)["unseen_accuracy"]}
        c["alpha0_expressivity"] = report["domains"][k]["alpha0_expressivity"]
        c["partition_function_check"] = report["domains"][k]["partition_function_check"]
        checks[k] = c
    report["sanity_checks"] = checks
    report["runtime_s"] = time.time() - t0
    write_json(os.path.join(od, "report.json"), report)
    md = render_markdown(report, table, agg, sens, cfg)
    with open(os.path.join(od, "RECONSTRUCTION_REPORT.md"), "w") as f:
        f.write(md)
    write_json(os.path.join(od, "sanity_checks.json"), checks)
    print(md)


def _seen_acc(spec, h):
    ev = evaluate_hypothesis(spec, h)
    seen = [t for t in spec.heldout if t.name not in spec.unseen_names]
    return float(np.mean([ev["predictions"][t.name] == t.label for t in seen]))


def render_markdown(report, table, agg, sens, cfg) -> str:
    L = ["# Reconstruction report", ""]
    L += ["Hyper-parameters: beta=%g, lambda=%g, gamma=%g, W={0,%g,...,%g}, rho=%g (min running baseline)." % (
        cfg["beta"], cfg["lam"], cfg["gamma"], cfg["w_step"], cfg["w_max"], cfg["rho"]), ""]
    for k, D in report["domains"].items():
        L += [f"## Domain `{k}`", "", "```"] + D["layout_ascii"] + ["```", D["layout_legend"], ""]
        L += [f"Items: {D['items']}", f"Mechanics: {D['mechanics']}", f"Vocabulary levels: {D['vocabulary_levels']}",
              f"Hidden norm: {D['hidden_norm']}", "", "### Training demonstrations", ""]
        for t in D["training"]:
            L.append(f"- **{t['name']}** items={t['items']} len={t['length']}: `{t['compact']}`")
        L += ["", "### Held-out trajectories", "", "| name | label | unseen | len | actions |", "|---|---|---|---|---|"]
        for t in D["heldout"]:
            L.append(f"| {t['name']} | {'violating' if t['label_violating'] else 'compliant'} | "
                     f"{'yes' if t['unseen_structure'] else ''} | {t['length']} | `{t['compact']}` |")
        L += ["", "### Candidate hypotheses (level 0 from D1; level 1 from D1..D3)", "",
              "| id | lvl | intended | L(α) | L(N) | prior | V on D1..D3 | held-out acc if MAP | norm | definitions |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for h in D["candidate_hypotheses"]:
            L.append(f"| {h['id']} | {h['level']} | {'✓' if h['intended'] else ''} | {h['L_alpha']} | {h['L_N']} | "
                     f"{h['prior_joint']:.3f} | {h['violations_on_training']} | {h['heldout_accuracy_if_selected']:.2f} | "
                     f"{h['norm']} | {'; '.join(h['definitions'])} |")
        L += ["", "Partition-function check (fixed point vs direct sparse solve): " +
              ", ".join(f"w={c['w']}: {c['logZ_fixed_point']:.6f} vs {c['logZ_direct']:.6f}"
                        for c in D["partition_function_check"]), ""]
    L += ["## Original protocol (3 demonstrations, deterministic)", "",
          "| domain | method | acc | viol. F1 | unseen acc | MAP | MAP intended | P(intended) | refinements |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in table:
        L.append(f"| {r['domain']} | {r['method']} | {r['accuracy']:.2f} | {r['violation_f1']:.2f} | "
                 f"{'' if r.get('unseen_accuracy') is None else '%.2f' % r['unseen_accuracy']} | {r.get('map_id', '')} | "
                 f"{r.get('map_is_intended', '')} | {'' if r.get('P_intended') is None else '%.3f' % r['P_intended']} | "
                 f"{r.get('n_refinements', '')} |")
    L += ["", "## Supplement: same tasks, optimal demonstrations with random tie-breaking (mean, 95% bootstrap CI)", "",
          "| domain | method | n | acc | viol. F1 | unseen acc | MAP=intended rate | refined rate |", "|---|---|---|---|---|---|---|---|"]
    for r in agg:
        def f(m):
            return "" if f"{m}_mean" not in r else f"{r[m + '_mean']:.2f} [{r[m + '_ci_lo']:.2f},{r[m + '_ci_hi']:.2f}]"
        L.append(f"| {r['domain']} | {r['method']} | {r['n_seeds']} | {f('accuracy')} | {f('violation_f1')} | "
                 f"{f('unseen_accuracy')} | {r.get('rate_map_intended', '')} | {r.get('rate_refined', '')} |")
    L += ["", "## MLCI threshold sensitivity (supplementary; the primary threshold was fixed a priori)", "",
          "| domain | epsilon | acc | F1 | unseen | #constraints |", "|---|---|---|---|---|---|"]
    for s in sens:
        L.append(f"| {s['domain']} | {s['epsilon']} ({s['epsilon_used']:.2f}) | {s['accuracy']:.2f} | "
                 f"{s['violation_f1']:.2f} | {s['unseen_accuracy']} | {s['n_constraints']} |")
    L += ["", "## Sanity checks", "", "```json", json.dumps(report["sanity_checks"], indent=1, default=str), "```"]
    return "\n".join(L)


if __name__ == "__main__":
    main()
