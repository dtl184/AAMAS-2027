"""NEW EXPERIMENT 3: identifiability and disambiguating demonstrations (cart-use domain).

Hypothesis set: the oracle set (alpha_0 norms proposed from D1 + all refined-level hypotheses), held fixed
so that identifiability is studied separately from refinement triggering.

Key fact used throughout.  Under the max-ent model, for two hypotheses whose violation counts are binary
(V in {0,1}; true for every cart hypothesis), the likelihood ratio of a demonstration depends on the
trajectory only through (V1(tau), V2(tau)):
    P(tau | h_i, D) = exp(-beta C(tau)) * sum_w P(w | h_i, D) exp(-beta w V_i(tau)) / Z_i(w).
Hence (i) a demonstration that is compliant under both hypotheses carries evidence only through its *task
context* (s0, g) -- never through which compliant path was taken -- and (ii) the exact expected information
a demonstration in context c carries about {h1, h2} is computable from the four path masses
    N_ab(c) = sum_{tau: V1 = a, V2 = b} exp(-beta C(tau)),
obtained from one linear solve on the joint product of both monitors.  The diagnostic-demonstration search
picks the context maximising the mutual information I(H; tau | c) between the two currently most probable
hypotheses (no knowledge of the true norm is used); the true demonstrator then demonstrates in it.

Sequences (all demonstrations are optimal under the hidden norm):
  diagnostic : D1-D3 (original), D4-D5 ordinary, D6 actively selected, D7-D8 ordinary
  control    : D1-D3, D4-D8 ordinary (redundant, non-diagnostic)
  reacquire  : D1-D5, D6 = compliant 'return, re-acquire, return again' demonstration, D7-D8 ordinary
plus: one extra demonstration after D1-D5 chosen (a) actively, (b) uniformly from all contexts,
(c) uniformly from the standard (ordinary) contexts; and a long control with 20 random ordinary demos.

python -m experiments.identifiability [--quick]
"""
from __future__ import annotations

import itertools
import os
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl
from scipy.special import logsumexp

from abstractions.base import Batch
from analysis import plotting as P
from environment.domains import get_spec
from environment.shopping_domain import CART_STATION
from environment.trajectories import ScriptBuilder, Trajectory, plan_optimal
from experiments.common import (FIGURES, bootstrap_ci, make_zc, out_dir, parse_args, pmap, run_metadata, seed_for,
                                write_csv, write_json, write_jsonl)
from inference.posterior import Posterior
from inference.predictive import classify, evaluate_predictions
from norms.hypotheses import Hypothesis
from proposals.deterministic import DeterministicProposalProvider


# ----------------------------------------------------------------------------------------- hypotheses
def hypothesis_set(spec):
    prov = DeterministicProposalProvider()
    d = prov.propose_initial(spec, spec.training[:1]) + prov.refine(spec, spec.training[:1], [], 1)
    hs = [Hypothesis.from_dict(x) for x in d]
    for h in hs:
        h.validate(spec.domain.vocab_levels[h.level])
    return hs


# ----------------------------------------------------------------------------------------- contexts
def context_pool(spec, max_items: int):
    d = spec.domain
    L = d.layout
    carts = [("station", CART_STATION)] + [(f"parked{c}", L.cell_index[c]) for c in L.free_cells]
    out = []
    for k in range(max_items + 1):
        for items in itertools.combinations(d.item_names, k):
            for cname, cart in carts:
                out.append({"items": items, "cart": cname,
                            "task": d.make_task(items, (d.entrance_idx, cart, 0, 0))})
    return out


def standard_contexts(spec):
    return [c for c in context_pool(spec, 3) if c["cart"] == "station" and len(c["items"]) >= 1]


# ----------------------------------------------------------------------------------------- exact pair masses
def pair_log_masses(spec, task, h1: Hypothesis, h2: Hypothesis, beta: float) -> np.ndarray:
    """log N_ab for a, b in {0,1}: path mass of trajectories with (V1, V2) = (a, b)."""
    d = spec.domain
    g = d.graph(task)
    batch = Batch.from_graph(d, g)
    K1, i1, n1, v1 = h1.norm.monitor(h1.abstraction, batch)
    K2, i2, n2, v2 = h2.norm.monitor(h2.abstraction, batch)
    K = K1 * K2
    n = g.n_states * K
    wgt = np.exp(-beta)
    rows, cols = [], []
    B = np.zeros((n, 4))
    term = g.E_dst < 0
    for m1 in range(K1):
        for m2 in range(K2):
            if np.any(v1[m1][~term]) or np.any(v2[m2][~term]):
                raise ValueError("pair masses require violations only on terminal transitions (binary V)")
            src = g.E_src * K + m1 * K2 + m2
            nt = ~term
            rows.append(src[nt])
            cols.append(g.E_dst[nt] * K + n1[m1][nt] * K2 + n2[m2][nt])
            cat = 2 * v1[m1][term] + v2[m2][term]
            np.add.at(B, (src[term], cat), wgt)
    r, c = np.concatenate(rows), np.concatenate(cols)
    M = sp.csr_matrix((np.full(len(r), wgt), (r, c)), shape=(n, n))
    z = spl.splu((sp.identity(n, format="csc") - M).tocsc()).solve(B)
    start = 0 * K + i1 * K2 + i2
    with np.errstate(divide="ignore"):
        return np.log(np.maximum(z[start], 0.0)).reshape(2, 2)


def pair_information(lj1: np.ndarray, lj2: np.ndarray, W: np.ndarray, logN: np.ndarray, beta: float,
                     equal_prior: bool = False) -> dict:
    """Exact mutual information (nats) between H in {h1, h2} and the next demonstration in a context.

    lj1, lj2: rows of the (unnormalised or normalised) log posterior log P(h_i, w | D) over W.
    equal_prior=True scores the context as if the two hypotheses were equally probable (landscape view).
    """
    lm = np.array([logsumexp(lj1), logsumexp(lj2)])
    pi = np.array([0.5, 0.5]) if equal_prior else np.exp(lm - logsumexp(lm))
    probs = []
    for which, lj in enumerate((lj1, lj2)):
        pw = np.exp(lj - logsumexp(lj))                          # P(w | h_i, D)
        V = np.array([[0, 0], [1, 1]]) if which == 0 else np.array([[0, 1], [0, 1]])
        # log P_{h_i,w}(a,b) = -beta w V_i(ab) + log N_ab - log Z_i(w)
        lt = -beta * W[:, None, None] * V[None] + logN[None]
        lZ = logsumexp(lt.reshape(len(W), -1), axis=1)
        probs.append(np.einsum("w,wab->ab", pw, np.exp(lt - lZ[:, None, None])))
    mix = pi[0] * probs[0] + pi[1] * probs[1]
    mi = 0.0
    for k in range(2):
        m = probs[k] > 0
        mi += pi[k] * np.sum(probs[k][m] * (np.log(probs[k][m]) - np.log(mix[m])))
    with np.errstate(divide="ignore", invalid="ignore"):
        lbf = np.log(probs[0]) - np.log(probs[1])
    ebf = float(np.nansum(mix * np.abs(np.where(np.isfinite(lbf), lbf, 0.0))))
    return {"mutual_information": float(mi), "expected_abs_log_bf": ebf, "P_V_h1": probs[0].tolist(),
            "P_V_h2": probs[1].tolist()}


# ----------------------------------------------------------------------------------------- sequences
def true_demo(spec, task, name, rng=None) -> Trajectory:
    t = plan_optimal(spec.domain, task, spec.true_hypothesis, rng=rng, name=name)
    t.label = False
    return t


def reacquire_demo(spec, items, name) -> Trajectory:
    L = spec.domain.layout
    b = ScriptBuilder(spec.domain, spec.domain.make_task(items)).goto(L.cart_station).pickup_cart()
    for it in items:
        b.pick(it)
    t = b.goto(L.return_area).leave_cart().pickup_cart().leave_cart().exit().build(name, False, kind="reacquire_compliant")
    assert not spec.violates(t)
    return t


ORDINARY = [("eggs", "milk"), ("apple", "bread"), ("bread", "milk"), ("apple", "eggs", "milk"), ("apple", "eggs")]


def _score_context(job):
    ctx, hd1, hd2, lj1, lj2, W, beta, equal_prior = job
    spec = get_spec("cart")
    logN = pair_log_masses(spec, ctx["task"], Hypothesis.from_dict(hd1), Hypothesis.from_dict(hd2), beta)
    info = pair_information(np.asarray(lj1), np.asarray(lj2), np.asarray(W), logN, beta, equal_prior)
    return {"items": "+".join(ctx["items"]) or "-", "cart": ctx["cart"], **info, "logN": logN.tolist()}


def score_contexts(post, i1, i2, pool, cfg, equal_prior=False):
    lj = post.log_joint()
    h1, h2 = post.hyps[i1].to_dict(), post.hyps[i2].to_dict()
    jobs = [(c, h1, h2, lj[i1], lj[i2], post.W, cfg["beta"], equal_prior) for c in pool]
    return pmap(_score_context, jobs, cfg["workers"])


class _PostState:
    """Picklable recipe to rebuild a posterior inside a worker (hypothesis set + demonstrations)."""

    def __init__(self, cfg, demos):
        self.cfg, self.demos = cfg, demos

    def __call__(self):
        spec = get_spec("cart")
        post = Posterior(make_zc(spec, self.cfg), self.cfg["lam"], self.cfg["gamma"], self.cfg["prior_mode"],
                         self.cfg["dl_coding"])
        for t in self.demos:
            post.add_demo(t)
        post.add_hypotheses(hypothesis_set(spec))
        return post


def make_posterior(spec, cfg, demos):
    return _PostState(cfg, demos)()


def select_diagnostic(spec, cfg, demos, pool, exclude_alt=None):
    """Context maximising the exact mutual information between the two most probable hypotheses."""
    post = make_posterior(spec, cfg, demos)
    m = post.marginal()
    order = np.argsort(-m)
    i1, i2 = int(order[0]), int(order[1])
    hids = [h.hid for h in post.hyps]
    scores = score_contexts(post, i1, i2, pool, cfg)
    best = int(np.argmax([s["mutual_information"] for s in scores]))
    return pool[best], scores, (hids[i1], hids[i2])


def prefix_records(spec, cfg, demos, label, intended_key, alt_id, extra=None):
    post = make_posterior(spec, cfg, demos)
    hids = [h.hid for h in post.hyps]
    i_int = post.index_of_key(intended_key)
    i_alt = hids.index(alt_id)
    recs = []
    W = post.W
    for t in range(0, len(demos) + 1):
        m = post.marginal(upto=t)
        lj = post.log_joint(upto=t)
        lm = logsumexp(lj, axis=1)
        # likelihood-only Bayes factor (w integrated under its uniform prior)
        ll = [np.sum(post.loglik[i][:t], axis=0) if t else np.zeros(len(W)) for i in (i_int, i_alt)]
        lbf = float(logsumexp(ll[0]) - logsumexp(ll[1]))
        imap = int(np.argmax(m))
        ev = evaluate_predictions(spec, classify(spec.domain, post.hyps[imap], spec.heldout))
        recs.append({"sequence": label, "t": t, "demo": demos[t - 1].name if t else None,
                     "demo_context": (("+".join(demos[t - 1].task.items) or "-") + " | cart=" +
                                      _cart_name(spec, demos[t - 1].task)) if t else None,
                     "P_intended": float(m[i_int]), "P_alternative": float(m[i_alt]),
                     "P_others": float(1 - m[i_int] - m[i_alt]), "entropy": post.entropy(upto=t),
                     "log_posterior_odds_int_vs_alt": float(lm[i_int] - lm[i_alt]),
                     "log_bayes_factor_int_vs_alt": lbf, "map_id": hids[imap],
                     "accuracy": ev["accuracy"], "violation_f1": ev["violation_f1"], **(extra or {})})
    return recs, post


def _cart_name(spec, task):
    c = task.start[1]
    return "station" if c == CART_STATION else ("held" if c < 0 else str(spec.domain.cell(c)))


def _random_extra(job):
    kind, s, cfg, base_demos, intended_key, alt_id = job
    spec = get_spec("cart")
    rng = np.random.default_rng(seed_for(cfg, "ident", kind, s))
    pool = context_pool(spec, cfg["ident_max_items"]) if kind == "random_any" else standard_contexts(spec)
    c = pool[int(rng.integers(len(pool)))]
    t = true_demo(spec, c["task"], "D6", rng=rng)
    recs, _ = prefix_records(spec, cfg, base_demos + [t], kind, intended_key, alt_id)
    r = recs[-1]
    r.update({"seed": s, "context_items": "+".join(c["items"]) or "-", "context_cart": c["cart"]})
    return r


def _long_control(job):
    s, cfg, intended_key, alt_id, n_extra = job
    spec = get_spec("cart")
    rng = np.random.default_rng(seed_for(cfg, "ident_long", s))
    pool = standard_contexts(spec)
    demos = list(spec.training)
    for k in range(n_extra):
        c = pool[int(rng.integers(len(pool)))]
        demos.append(true_demo(spec, c["task"], f"D{len(demos) + 1}", rng=rng))
    recs, _ = prefix_records(spec, cfg, demos, "long_control", intended_key, alt_id, extra={"seed": s})
    return recs


# ----------------------------------------------------------------------------------------- main
def main():
    cfg = parse_args("Identifiability / disambiguating demonstrations")
    od = out_dir(cfg, "identifiability")
    write_json(os.path.join(od, "metadata.json"), run_metadata("identifiability", cfg))
    t0 = time.time()
    spec = get_spec("cart")
    d = spec.domain
    intended_key = spec.true_hypothesis.key
    hyps = hypothesis_set(spec)
    pool = context_pool(spec, cfg["ident_max_items"])
    print(f"{len(hyps)} hypotheses, {len(pool)} candidate contexts")
    # warm the partition-function cache for every (hypothesis, context) in parallel
    zc_jobs = [(h.to_dict(), c["task"]) for c in pool + standard_contexts(spec) for h in hyps]
    pmap(_warm, [(j, cfg) for j in zc_jobs], cfg["workers"])

    ordinary = [true_demo(spec, d.make_task(it), f"D{4 + k}") for k, it in enumerate(ORDINARY)]
    base5 = list(spec.training) + ordinary[:2]
    # --- main competitor: most probable non-intended hypothesis after the 5 non-diagnostic demonstrations
    post5 = make_posterior(spec, cfg, base5)
    m5 = post5.marginal()
    order = [i for i in np.argsort(-m5) if post5.hyps[i].key != intended_key]
    alt_id = post5.hyps[order[0]].hid
    print("main competitor after D1-D5:", alt_id, post5.table(top=5))

    # --- active selection (two rounds)
    c6, scores6, pair6 = select_diagnostic(spec, cfg, base5, pool)
    D6 = true_demo(spec, c6["task"], "D6")
    D6.meta.update({"kind": "diagnostic", "selected_for_pair": pair6})
    D7 = true_demo(spec, d.make_task(ORDINARY[2]), "D7")
    D8 = true_demo(spec, d.make_task(ORDINARY[3]), "D8")
    seq_diag = base5 + [D6, D7, D8]
    ctrl = list(spec.training) + [true_demo(spec, d.make_task(it), f"D{4 + k}") for k, it in enumerate(ORDINARY)]
    reacq = base5 + [reacquire_demo(spec, ("apple", "milk"), "D6_reacquire"), ordinary[2], ordinary[3]]

    recs = []
    for label, seq in (("diagnostic", seq_diag), ("control", ctrl), ("reacquire", reacq)):
        r, _ = prefix_records(spec, cfg, seq, label, intended_key, alt_id)
        recs += r
    write_csv(os.path.join(od, "prefix_posteriors.csv"), recs)
    write_jsonl(os.path.join(od, "sequences.jsonl"),
                [{"sequence": lab, "demos": [t.to_dict(d) for t in seq]}
                 for lab, seq in (("diagnostic", seq_diag), ("control", ctrl), ("reacquire", reacq))])
    write_csv(os.path.join(od, "context_scores_round1.csv"), sorted(scores6, key=lambda s: -s["mutual_information"]))

    # --- MI landscape: intended vs every competitor, over all candidate contexts (given D1-D5)
    hids = [h.hid for h in post5.hyps]
    i_int = post5.index_of_key(intended_key)
    land = []
    for j, h in enumerate(post5.hyps):
        if j == i_int or h.hid == "H_null":
            continue
        sc = score_contexts(post5, i_int, j, pool, cfg, equal_prior=True)
        best = max(sc, key=lambda s: s["mutual_information"])
        std = [s for s in sc if s["cart"] == "station" and s["items"] != "-"]
        best_std = max(std, key=lambda s: s["mutual_information"])
        land.append({"competitor": h.hid, "MI_scored_with": "equal pair prior, w-posteriors given D1-D5", "competitor_posterior_after_D5": float(m5[j]),
                     "best_context": f"{best['items']} | cart={best['cart']}", "best_MI": best["mutual_information"],
                     "best_expected_abs_logBF": best["expected_abs_log_bf"],
                     "best_standard_context": f"{best_std['items']} | cart=station",
                     "best_standard_MI": best_std["mutual_information"],
                     "median_MI_all_contexts": float(np.median([s["mutual_information"] for s in sc]))})
    write_csv(os.path.join(od, "mi_landscape.csv"), land)

    # --- random vs diagnostic extra demonstration (after D1-D5)
    diag_rec, _ = prefix_records(spec, cfg, base5 + [D6], "active", intended_key, alt_id)
    extra = [dict(diag_rec[-1], seed=None, context_items="+".join(c6["items"]) or "-", context_cart=c6["cart"])]
    jobs = [(kind, s, cfg, base5, intended_key, alt_id) for kind in ("random_any", "random_standard")
            for s in range(cfg["ident_random_seeds"])]
    extra += pmap(_random_extra, jobs, cfg["workers"])
    write_csv(os.path.join(od, "extra_demo_comparison.csv"), extra)
    comp = []
    for kind in ("active", "random_any", "random_standard"):
        rs = [r for r in extra if r["sequence"] == kind]
        row = {"selection": kind, "n": len(rs)}
        for mtr in ("P_intended", "log_posterior_odds_int_vs_alt", "accuracy", "violation_f1"):
            mu, lo, hi = bootstrap_ci([r[mtr] for r in rs], cfg["bootstrap"])
            row.update({f"{mtr}_mean": mu, f"{mtr}_ci_lo": lo, f"{mtr}_ci_hi": hi})
        row["fraction_MAP_intended"] = float(np.mean([r["map_id"] == "H_int" for r in rs]))
        comp.append(row)
    write_csv(os.path.join(od, "extra_demo_summary.csv"), comp)

    # --- long control: up to 20 additional random ordinary demonstrations
    n_long = 20 if not cfg["_args"].get("quick") else 4
    longr = pmap(_long_control, [(s, cfg, intended_key, alt_id, n_long) for s in range(min(10, cfg["ident_random_seeds"]))],
                 cfg["workers"])
    longr = [r for rs in longr for r in rs]
    write_csv(os.path.join(od, "long_control.csv"), longr)

    summary = {"hypotheses": [{"id": h.hid, "description": h.describe(), "L_alpha": h.L_alpha(), "L_norm": h.L_norm()}
                              for h in post5.hyps],
               "intended": "H_int", "main_competitor_after_D5": alt_id,
               "posterior_after_D5": post5.table(top=6),
               "round1": {"pair": pair6, "context": {"items": c6["items"], "cart": c6["cart"]},
                          "demo": D6.to_dict(d), "best_scores": sorted(scores6, key=lambda s: -s["mutual_information"])[:5]},
               "mi_landscape": land, "extra_demo_summary": comp,
               "final_prefix": {lab: next(r for r in reversed(recs) if r["sequence"] == lab)
                                for lab in ("diagnostic", "control", "reacquire")},
               "consistency_check_logZ": _consistency(spec, cfg, hyps, c6),
               "runtime_s": time.time() - t0}
    write_json(os.path.join(od, "summary.json"), summary)
    figs = plots(recs, extra, longr, land, alt_id, c6)
    write_json(os.path.join(od, "outputs.json"), {"figures": figs})
    print(json.dumps({k: summary[k] for k in ("main_competitor_after_D5", "round1", "extra_demo_summary",
                                              "final_prefix", "consistency_check_logZ")}, indent=1, default=str)[:6000])
    print(f"done in {time.time() - t0:.0f}s")


def _warm(job):
    (hd, task), cfg = job
    spec = get_spec("cart")
    make_zc(spec, cfg).logZ(Hypothesis.from_dict(hd), task)
    return None


def _consistency(spec, cfg, hyps, ctx):
    """Check sum_ab N_ab e^{-beta w V_i} reproduces log Z_i(w) from the main solver."""
    zc = make_zc(spec, cfg)
    h1, h2 = hyps[0], next(h for h in hyps if h.key == spec.true_hypothesis.key)
    logN = pair_log_masses(spec, ctx["task"], h1, h2, cfg["beta"])
    out = []
    for j in (0, 20, 60):
        w = zc.W[j]
        for which, h in ((0, h1), (1, h2)):
            V = np.array([[0, 0], [1, 1]]) if which == 0 else np.array([[0, 1], [0, 1]])
            lz = float(logsumexp(-cfg["beta"] * w * V + logN))
            out.append({"h": h.hid, "w": float(w), "from_pair_masses": lz, "from_ZComputer": float(zc.logZ(h, ctx["task"])[j])})
    return out


def plots(recs, extra, longr, land, alt_id, c6):
    P.setup()
    paths = []
    fig, axes = P.plt.subplots(1, 3, figsize=(13.5, 3.6), sharey=True)
    titles = {"diagnostic": "Diagnostic sequence", "control": "Control: redundant ordinary demos",
              "reacquire": "Intuitive 'return–reacquire–return' demo"}
    for ax, lab in zip(axes, ("diagnostic", "control", "reacquire")):
        rs = [r for r in recs if r["sequence"] == lab]
        t = [r["t"] for r in rs]
        ax.plot(t, [r["P_intended"] for r in rs], color=P.SLOTS[0], marker="o", ms=4, label="intended: H_int")
        ax.plot(t, [r["P_alternative"] for r in rs], color=P.SLOTS[1], marker="s", ms=4, label=f"main competitor: {alt_id}")
        ax.plot(t, [r["P_others"] for r in rs], color=P.INK2, ls="--", lw=1.2, label="all other hypotheses")
        if lab == "diagnostic":
            ax.axvline(6, color=P.INK, ls="--", lw=0.9)
            ax.text(6, 1.02, f"D6 (diagnostic): {'+'.join(c6['items']) or 'no items'},\ncart {c6['cart']}",
                    fontsize=6.5, ha="center", va="bottom")
        if lab == "reacquire":
            ax.axvline(6, color=P.INK, ls="--", lw=0.9)
            ax.text(6, 1.02, "D6: return, reacquire,\nreturn again", fontsize=6.5, ha="center", va="bottom")
        ax.set_xlabel("number of demonstrations observed")
        ax.set_title(titles[lab], pad=22)
        ax.set_xticks(t)
        ax.set_ylim(-0.02, 1.0)
    axes[0].set_ylabel("posterior probability")
    axes[0].legend(fontsize=7, loc="center left")
    p_ = os.path.join(FIGURES, "identifiability_posterior.png")
    P.save(fig, p_)
    paths.append(p_)
    # extra-demo comparison + long control
    fig, axes = P.plt.subplots(1, 2, figsize=(10, 3.4))
    ax = axes[0]
    for k, (kind, lab) in enumerate((("active", "actively selected"), ("random_any", "random context"),
                                     ("random_standard", "random ordinary context"))):
        v = [r["P_intended"] for r in extra if r["sequence"] == kind]
        x = np.full(len(v), k) + np.random.default_rng(0).uniform(-0.15, 0.15, len(v)) * (len(v) > 1)
        ax.scatter(x, v, s=14, color=P.SLOTS[k], alpha=0.7, edgecolor="none")
        ax.plot([k - 0.25, k + 0.25], [np.mean(v)] * 2, color=P.INK, lw=2)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["actively selected", "random context", "random ordinary"], fontsize=8)
    ax.set_ylabel("P(intended) after one extra demo")
    ax.set_title("One extra demonstration after D1–D5")
    ax.set_ylim(-0.02, 1.02)
    ax = axes[1]
    seeds = sorted({r["seed"] for r in longr})
    for s in seeds:
        rs = [r for r in longr if r["seed"] == s]
        ax.plot([r["t"] for r in rs], [r["P_intended"] for r in rs], color=P.SLOTS[0], alpha=0.35, lw=1)
    ts = sorted({r["t"] for r in longr})
    ax.plot(ts, [np.mean([r["P_intended"] for r in longr if r["t"] == t]) for t in ts], color=P.SLOTS[0], lw=2.2,
            label="P(intended), mean over seeds")
    ax.plot(ts, [np.mean([r["P_alternative"] for r in longr if r["t"] == t]) for t in ts], color=P.SLOTS[1], lw=2.2,
            label=f"P({alt_id}), mean")
    ax.set_xlabel("number of demonstrations (D1–D3 + random ordinary)")
    ax.set_ylabel("posterior probability")
    ax.set_title("Long control: ordinary demonstrations only")
    ax.legend(fontsize=7)
    p_ = os.path.join(FIGURES, "identifiability_random_vs_diagnostic.png")
    P.save(fig, p_)
    paths.append(p_)
    return paths


import json  # noqa: E402

if __name__ == "__main__":
    main()
