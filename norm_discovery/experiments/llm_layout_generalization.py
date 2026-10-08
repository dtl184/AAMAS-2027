"""Can GPT-5.5 propose a layout-independent norm instead of a layout-specific (coordinate) rule?

Spatial (aisle) domain only.  Three store layouts built with the existing Layout / ShoppingDomain machinery:
  L1  the existing Exp. 2 store (vertical dead-end aisles at columns 3/6/9, rows 0-2)          -- training
  L2  a second store with horizontal aisles at a different position                               -- training (c3 only)
  L3  a third store, never shown to any LLM: one vertical aisle at a new column and one horizontal aisle at a new
      row; cells that were aisle cells in L1/L2 are free non-aisle cells here (coordinate "traps")  -- test only
The hidden norm is identical everywhere: FORBIDDEN hasCart ∧ inAisle, inAisle := (shelfW∧shelfE) ∨ (shelfN∧shelfS).

Conditions (10 independent GPT-5.5 calls each; replicate r pairs the same initial hypothesis set across conditions):
  c1  single layout, no generalisation guidance: the EXISTING refinement prompt and responses (cached GPT-5.5
      runs orig/aisle/r of the extended threshold study; no new call)
  c2  single layout + generalisation guidance: c1's exact prompt with one guidance paragraph inserted
  c3  two layouts (L1 + L2 demonstrations) + the same guidance
Stages:  prepare (no API) -> run (20 new calls; aborts if OPENAI_API_KEY missing) -> analyze (no API)
    bash -lc 'source ~/.bashrc && cd /home/train/norm_discovery && /home/train/anaconda3/bin/python -m experiments.llm_layout_generalization run'
"""
from __future__ import annotations

import collections
import json
import os
import sys
import time

import numpy as np
from scipy.special import logsumexp

from abstractions.base import atom_family
from environment.domains import MLCI_FAMILIES, DomainSpec, get_spec
from environment.gridworld import Layout
from environment.shopping_domain import ShoppingDomain
from environment.trajectories import ScriptBuilder, Trajectory, is_valid, plan_optimal
from experiments.common import (DEFAULTS, RESULTS, bootstrap_ci, make_provider, parse_args, pmap, run_metadata,
                                write_csv, write_json, write_jsonl)
from inference.likelihood import ZComputer
from inference.predictive import metrics
from norms.hypotheses import Hypothesis
from norms.violations import count_violations
from proposals import templates as T
from proposals.base import render_label

OUT = os.path.join(RESULTS, "llm_layout_generalization")
EXT = os.path.join(RESULTS, "gpt55", "refinement_threshold_extended", "aisle", "rho_20000")
N_RUNS = 10
VOCAB = {0: ("coords", "cart_possession", "actions_coarse"), 1: ("coords", "cart_possession", "actions_coarse", "geometry")}

GUIDANCE = (
    "Generalisation requirement. Some candidate rules merely describe the particular store in which the "
    "demonstrations were recorded -- for example rules that mention specific coordinates, cells or regions of this "
    "floor plan. Such rules can fit the demonstrations without capturing the social norm, and would be meaningless "
    "in a store with a different floor plan. Distinguish (a) candidate norms that express general behavioural "
    "constraints of the shopping domain from (b) rules that only describe the geometry or coordinates of this "
    "particular environment. Propose candidates of type (a): define any abstractions from layout-independent "
    "relations rather than from coordinates (at(x,y)), so that the same rule could be applied unchanged in any "
    "store layout. Every candidate must still be consistent with all demonstrations. Add a field \"justification\" "
    "(one sentence) to each element explaining why the rule remains meaningful across different store layouts.")

# ----------------------------------------------------------------------------------------------- layouts
L2_ROWS = [
    "............",
    "...######...",
    "...######...",
    "............",          # horizontal aisle: cells (3..8, 3)
    "...######...",
    "...######...",
    "............",
    "..........#.",          # standalone display (10,7)
    "............",
    "EC.........X",
]
L2_ITEMS = {"b1": (4, 2), "b2": (7, 4), "b3": (10, 7)}
L2_AISLES = {"H": [(x, 3) for x in range(3, 9)]}

L3_ROWS = [
    "............",
    "#.##........",          # vertical aisle: cells (1, 1..3)
    "#.##........",
    "#.##........",
    "..........#.",          # standalone display (10,4)
    "............",
    "..########..",
    "..########..",
    "............",          # horizontal aisle: cells (2..9, 8)
    "..########..",
    "..########..",
    "............",
    "EC.........X",
]
L3_ITEMS = {"v": (2, 2), "h": (3, 7), "s": (10, 4)}
L3_AISLES = {"V": [(1, y) for y in (1, 2, 3)], "H": [(x, 8) for x in range(2, 10)]}


def intended():
    h = Hypothesis.from_dict(next(x for x in T.exp2_level1() if x["id"] == "S_inAisle"))
    h.hid = "TRUE"
    return h


def make_domain(name, rows, items, aisles):
    lay = Layout.from_ascii(name, rows, items, aisles=aisles)
    return ShoppingDomain(name, lay, exit_requires_cart=True, vocab_levels=VOCAB, mlci_families=MLCI_FAMILIES)


def spec_L2():
    d = make_domain("genlayout_L2", L2_ROWS, L2_ITEMS, L2_AISLES)
    th = intended()
    train = [plan_optimal(d, d.make_task(it), th, name=f"B{i + 1}") for i, it in enumerate([("b1",), ("b2",), ("b3",)])]
    for t in train:
        t.label = False
    s = DomainSpec("genlayout_L2", d, th, "S_inAisle", train, [], [], [], [])
    _check(s, train)
    return s


def spec_L3():
    d = make_domain("genlayout_L3", L3_ROWS, L3_ITEMS, L3_AISLES)
    th = intended()
    av = [c for cs in L3_AISLES.values() for c in cs]
    H = []

    def sb(items):
        return ScriptBuilder(d, d.make_task(items)).goto(d.layout.cart_station).pickup_cart()
    # vertical aisle (1,1..3), mouth (1,4), pick v from (1,2)
    H.append(sb(["v"]).goto((1, 4), avoid=av).leave_cart().pick("v", (1, 2)).goto((1, 4)).pickup_cart()
             .exit(avoid=av).build("V_compliant", False, part="vertical"))
    H.append(sb(["v"]).goto((1, 4), avoid=av).pick("v", (1, 2)).exit().build("V_cart_in_aisle", True, part="vertical"))
    H.append(sb(["v"]).goto((1, 3)).leave_cart().pick("v", (1, 2)).goto((1, 3)).pickup_cart().exit()
             .build("V_cart_parked_inside", True, part="vertical"))
    # horizontal aisle (2..9, 8), mouth (1,8), pick h from (3,8)
    H.append(sb(["h"]).goto((1, 8), avoid=av).leave_cart().pick("h", (3, 8)).goto((1, 8)).pickup_cart()
             .exit(avoid=av).build("H_compliant", False, part="horizontal"))
    H.append(sb(["h"]).goto((1, 8), avoid=av).pick("h", (3, 8)).goto((1, 8)).exit(avoid=av)
             .build("H_cart_in_aisle", True, part="horizontal"))
    H.append(sb(["h"]).goto((10, 8), avoid=av).pick("h", (3, 8)).goto((1, 8)).exit(avoid=av)
             .build("H_cart_through_aisle", True, part="horizontal"))
    # standalone + coordinate traps (cart passes cells that were aisle cells in L1 / L2; compliant here)
    H.append(sb(["s"]).pick("s", avoid=av).exit(avoid=av).build("S_standalone_with_cart", False, part="standalone"))
    H.append(sb(["s"]).goto((0, 0), avoid=av).goto((3, 0), avoid=av).goto((9, 0), avoid=av).goto((9, 2), avoid=av)
             .pick("s", (10, 3)).exit(avoid=av).build("TRAP_L1_aisle_cells", False, part="trap"))
    H.append(sb(["s"]).goto((4, 4), avoid=av).goto((4, 3), avoid=av).goto((8, 3), avoid=av).pick("s", (10, 3))
             .exit(avoid=av).build("TRAP_L2_aisle_cells", False, part="trap"))
    b = sb(["v", "h"]).goto((1, 4), avoid=av).leave_cart().pick("v", (1, 2)).goto((1, 4)).pickup_cart()
    H.append(b.goto((1, 8), avoid=av).pick("h", (3, 8)).goto((1, 8)).exit(avoid=av)
             .build("VH_ok_then_cart_in_H", True, part="both"))
    s = DomainSpec("genlayout_L3", d, th, "S_inAisle", [], H, [t.name for t in H], [], [])
    _check(s, H)
    return s


def _check(spec, trajs):
    for t in trajs:
        ok, msg = is_valid(spec.domain, t)
        assert ok, (t.name, msg)
        assert (count_violations(spec.domain, spec.true_hypothesis, t) > 0) == bool(t.label), t.name


# ----------------------------------------------------------------------------------------------- spurious rules
def cartless_cells(domain, trajs):
    from proposals.deterministic import _cartless_excursions
    away, held = set(), set()
    for t in trajs:
        a, _, w = _cartless_excursions(domain, t)
        away |= a
        held |= w
    return sorted(away - held)


def spurious_hypotheses(layout_demos):
    """Layout-specific competitors: cart forbidden in (i) cells demonstrators visited without the cart, (ii) all aisle
    cells of the training layout(s) as a coordinate region.  Must be consistent with every training demonstration."""
    mem, region, held = set(), set(), set()
    for domain, demos in layout_demos:
        mem |= set(cartless_cells(domain, demos))
        region |= {c for cs in domain.layout.aisles.values() for c in cs}
        for t in demos:
            for s in t.states(domain):
                if s[1] == -1:
                    held.add(domain.cell(s[0]))
    region -= held
    mem -= held          # a coordinate rule cannot tell layouts apart: drop cells held with a cart anywhere
    out = []
    for hid, cells, txt in (("SPUR_memorised_cells", mem, "no cart in the cells demonstrators visited without it"),
                            ("SPUR_aisle_region_coords", region, "no cart in the training layout's aisle coordinates")):
        if cells:
            out.append(Hypothesis.from_dict({"id": hid, "level": 0, "template": "spurious", "text": txt, "abstraction": {},
                                             "norm": {"type": "prohibition",
                                                      "condition": {"and": ["hasCart", T.any_cell(sorted(cells))]}}}))
    for h in out:
        for domain, demos in layout_demos:
            assert all(count_violations(domain, h, t) == 0 for t in demos), f"{h.hid} inconsistent with training"
    return out


# ----------------------------------------------------------------------------------------------- prompts
def cached_c1(r):
    with open(os.path.join(EXT, f"full_run_{r:02d}", "transcript.json")) as f:
        tr = json.load(f)
    ini = next(p for p in tr["proposals"] if p["kind"] == "initial")
    ref = next(p for p in tr["proposals"] if p["kind"] == "refinement")
    with open(os.path.join(EXT, f"full_run_{r:02d}", "metrics.json")) as f:
        m = json.load(f)
    hyps = {h["level"]: [] for h in m["all_hypotheses"]}
    for h in m["all_hypotheses"]:
        hyps[h["level"]].append(h["json"])
    return {"prompt": ref["prompt"], "raw_response": ref["raw_response"], "initial": hyps[0], "refined": hyps[1],
            "model": [c.get("returned_model") for c in tr["llm_calls"]], "call": ref["attempts"][-1]["call"]}


def insert_guidance(prompt):
    key = "\nPropose 8 diverse candidate hypotheses."
    assert prompt.count(key) == 1
    return prompt.replace(key, "\n" + GUIDANCE + key)


def multi_layout_prompt(c2_prompt, l1, l1_demos, l2, l2_demos):
    """Replace the demonstration block by demonstrations from two layouts (everything else identical to c2)."""
    a = c2_prompt.index("Demonstration 1")
    b = c2_prompt.index("\n\nThe current hypotheses")
    fams = VOCAB[1]
    blocks = ["The demonstrations below come from TWO different stores (store A and store B) with different floor "
              "plans; coordinates refer to each store's own grid."]
    for tag, dom, demos in (("A", l1, l1_demos), ("B", l2, l2_demos)):
        for i, t in enumerate(demos):
            blocks.append(f"Store {tag}, demonstration {i + 1} (required items: {', '.join(t.task.items)}):\n" +
                          "\n".join(render_label(dom, t, fams)))
    return c2_prompt[:a] + "\n\n".join(blocks) + c2_prompt[b:]


# ----------------------------------------------------------------------------------------------- evaluation helpers
def classify_atoms(h):
    atoms = h.primitive_atoms()
    fams = {atom_family(a) for a in atoms}
    coord = bool(fams & {"coords", "cart_location"})
    geom = "geometry" in fams
    return "coordinate" if coord and not geom else ("mixed" if coord and geom else ("relational" if geom else "other"))


def multi_posterior(hyps, data, zcs, lam=0.2, gamma=0.2):
    """P(h | D) over demonstrations from several layouts (same priors / likelihood as the main experiments)."""
    W = next(iter(zcs.values())).W
    La = np.array([h.L_alpha() for h in hyps], float)
    Ln = np.array([h.L_norm() for h in hyps], float)
    lp = -lam * La - gamma * Ln
    lp = lp - logsumexp(lp)
    lj = np.tile(lp[:, None] - np.log(len(W)), (1, len(W)))
    for name, t in data:
        lj += np.array([zcs[name].loglik(h, t) for h in hyps])
    lj -= logsumexp(lj)
    return np.exp(logsumexp(lj, axis=1)), lj


def eval_on(spec, h, trajs):
    pred = [count_violations(spec.domain, h, t) > 0 for t in trajs]
    truth = [bool(t.label) for t in trajs]
    return metrics(pred, truth), pred


# ----------------------------------------------------------------------------------------------- stages
def setup():
    l1 = get_spec("aisle")
    l2, l3 = spec_L2(), spec_L3()
    return l1, l2, l3


def prepare(cfg, verbose=True):
    l1, l2, l3 = setup()
    os.makedirs(OUT, exist_ok=True)
    spur1 = spurious_hypotheses([(l1.domain, l1.training)])
    spur3 = spurious_hypotheses([(l1.domain, l1.training), (l2.domain, l2.training)])
    prompts = []
    for r in range(N_RUNS):
        c1 = cached_c1(r)
        p2 = insert_guidance(c1["prompt"])
        p3 = multi_layout_prompt(p2, l1.domain, l1.training, l2.domain, l2.training)
        for word in ("aisle", "Aisle", "inAisle", "narrow"):
            assert word not in GUIDANCE and word not in p3.split("Output format")[0].split("Available atoms")[0], word
        prompts.append({"run": r, "c1": c1["prompt"], "c2": p2, "c3": p3})
    layout_info = {k: {"ascii": s.domain.layout.ascii().split("\n"), "aisles": s.domain.layout.aisles,
                       "items": s.domain.layout.items} for k, s in (("L1", l1), ("L2", l2), ("L3", l3))}
    info = {"layouts": layout_info,
            "training": {"L1": [t.to_dict(l1.domain) for t in l1.training], "L2": [t.to_dict(l2.domain) for t in l2.training]},
            "L3_diagnostics": [{**t.to_dict(l3.domain), "part": t.meta["part"]} for t in l3.heldout],
            "spurious": {"single_layout": [h.to_dict() for h in spur1], "multi_layout": [h.to_dict() for h in spur3]},
            "guidance_paragraph": GUIDANCE,
            "held_constant": ["model gpt-5.5 and API-default generation settings (temperature unsupported)",
                              "for run r: the same initial (alpha_0) hypothesis set listed as 'current hypotheses'",
                              "prompt template, vocabulary, schema, 8 hypotheses requested",
                              "L1 demonstrations D1-D3 (all conditions)", "Bayesian priors / likelihood / W grid",
                              "evaluation sets (L1 held-out, L3 diagnostics) and probe sets"],
            "varies": {"c1 -> c2": "one guidance paragraph inserted", "c2 -> c3": "L2 demonstrations B1-B3 added and the "
                       "demonstration block is labelled store A / store B"}}
    write_json(os.path.join(OUT, "setup.json"), info)
    write_jsonl(os.path.join(OUT, "prompts.jsonl"), prompts)
    if verbose:
        for k, s in (("L1", l1), ("L2", l2), ("L3", l3)):
            print(k, "\n" + s.domain.layout.ascii())
        for t in l2.training:
            print("L2 train", t.name, t.pretty(l2.domain))
        for t in l3.heldout:
            print("L3 diag", t.name, t.label, t.pretty(l3.domain))
        print("spurious (single):", [h.describe() for h in spur1])
        print("spurious (multi):", [h.describe() for h in spur3])
        print(json.dumps({"planned_new_api_calls": 2 * N_RUNS, "c1_calls": "0 (cached)"}, indent=1))
    return l1, l2, l3, spur1, spur3, prompts


def _call(job):
    cfg, cond, r, prompt = job
    from proposals.openai_provider import CacheMissError
    prov = make_provider(cfg, replicate=f"genlayout/{cond}/{r}",
                         run_info={"experiment": "llm_layout_generalization", "condition": cond, "run_index": r})
    try:
        out = prov._ask(prompt, "refinement", 1, 3 if cond == "c2" else 6)
        err = None
    except Exception as e:
        out, err = [], f"{type(e).__name__}: {e}"
    comp = prov.complete
    rec = {"condition": cond, "run_index": r, "error": err, "proposals": out,
           "transcript": prov.transcript, "llm_calls": [{k: v for k, v in c.items() if k != "output_text"} for c in comp.calls]}
    d = os.path.join(OUT, "llm", cond, f"run_{r:02d}")
    os.makedirs(d, exist_ok=True)
    write_json(os.path.join(d, "transcript.json"), rec)
    return rec


def run(cfg):
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY is not set (source ~/.bashrc); no fallback provider is used.")
    l1, l2, l3, spur1, spur3, prompts = prepare(cfg, verbose=False)
    write_json(os.path.join(OUT, "metadata.json"), run_metadata("llm_layout_generalization", cfg))
    jobs = [(cfg, c, p["run"], p[c]) for p in prompts for c in ("c2", "c3")]
    res = pmap(_call, jobs, 8)
    usage = {"calls": sum(len(r["llm_calls"]) for r in res),
             "fresh": sum(1 for r in res for c in r["llm_calls"] if not c.get("cache_hit")),
             "input_tokens": sum((c.get("usage") or {}).get("input_tokens", 0) for r in res for c in r["llm_calls"] if not c.get("cache_hit")),
             "output_tokens": sum((c.get("usage") or {}).get("output_tokens", 0) for r in res for c in r["llm_calls"] if not c.get("cache_hit")),
             "models": sorted({c.get("returned_model") for r in res for c in r["llm_calls"] if c.get("returned_model")}),
             "errors": [(r["condition"], r["run_index"], r["error"][:200]) for r in res if r["error"]]}
    write_json(os.path.join(OUT, "api_usage.json"), usage)
    print(json.dumps(usage, indent=1))


def analyze(cfg):
    l1, l2, l3, spur1, spur3, prompts = prepare(cfg, verbose=False)
    specs = {"L1": l1, "L2": l2, "L3": l3}
    zcs = {k: ZComputer(s.domain, cfg["beta"], np.round(np.arange(0, cfg["w_max"] + 1e-9, cfg["w_step"]), 6),
                        cache_dir=os.path.join(RESULTS, "zcache")) for k, s in specs.items()}
    th = intended()
    rows, prop_rows = [], []
    for r in range(N_RUNS):
        c1 = cached_c1(r)
        for cond in ("c1", "c2", "c3"):
            if cond == "c1":
                dicts, err, model = c1["refined"], None, c1["model"]
                raw = c1["raw_response"]
            else:
                p = os.path.join(OUT, "llm", cond, f"run_{r:02d}", "transcript.json")
                if not os.path.exists(p):
                    rows.append({"condition": cond, "run_index": r, "status": "not run"})
                    continue
                with open(p) as f:
                    rec = json.load(f)
                dicts, err = rec["proposals"], rec["error"]
                model = [c.get("returned_model") for c in rec["llm_calls"]]
                if err:
                    rows.append({"condition": cond, "run_index": r, "status": "error", "error": err[:200]})
                    continue
            # validate: executable over the refined vocabulary AND evaluable in every layout
            refined, rejected = [], []
            for d in dicts:
                try:
                    h = Hypothesis.from_dict(d)
                    h.validate(VOCAB[1])
                    for s in specs.values():
                        for t in (s.training or s.heldout)[:1]:
                            count_violations(s.domain, h, t)
                    refined.append(h)
                except Exception as e:
                    rejected.append({"id": d.get("id"), "error": f"{type(e).__name__}: {e}"})
            for h in refined:
                eq = {k: s.equivalent(h, th) for k, s in specs.items()}
                m1 = eval_on(l1, h, l1.heldout)[0]
                m3 = eval_on(l3, h, l3.heldout)[0]
                prop_rows.append({"condition": cond, "run_index": r, "id": h.hid, "kind": classify_atoms(h),
                                  "equivalent_L1": eq["L1"], "equivalent_all_layouts": all(eq.values()),
                                  "L1_heldout_acc": m1["accuracy"], "L3_acc": m3["accuracy"], "L3_f1": m3["violation_f1"],
                                  "description": h.describe(), "justification": next((d.get("justification") for d in dicts
                                                                                     if d.get("id") == h.hid), None)})
            initial = [Hypothesis.from_dict(d) for d in c1["initial"]]
            spur = spur3 if cond == "c3" else spur1
            cand = initial + refined + spur
            data = [("L1", t) for t in l1.training] + ([("L2", t) for t in l2.training] if cond == "c3" else [])
            mpost, _ = multi_posterior(cand, data, zcs, cfg["lam"], cfg["gamma"])
            imap = int(np.argmax(mpost))
            hm = cand[imap]
            eq_all = [all(s.equivalent(h, th) for s in specs.values()) for h in cand]
            m1, _ = eval_on(l1, hm, l1.heldout)
            unseen = [t for t in l1.heldout if t.name in l1.unseen_names]
            mu = eval_on(l1, hm, unseen)[0]
            m3, pred3 = eval_on(l3, hm, l3.heldout)
            parts = {}
            for part in ("vertical", "horizontal", "trap", "standalone", "both"):
                idx = [i for i, t in enumerate(l3.heldout) if t.meta["part"] == part]
                if idx:
                    parts[part] = float(np.mean([pred3[i] == bool(l3.heldout[i].label) for i in idx]))
            kinds = collections.Counter(classify_atoms(h) for h in refined)
            rows.append({"condition": cond, "run_index": r, "status": "ok", "returned_models": sorted({x for x in model if x}),
                         "n_proposed": len(dicts), "n_valid": len(refined), "n_rejected": len(rejected),
                         "rejected": rejected, "n_relational": kinds["relational"], "n_coordinate": kinds["coordinate"],
                         "n_mixed": kinds["mixed"], "n_other": kinds["other"],
                         "frac_relational": kinds["relational"] / max(len(refined), 1),
                         "frac_coordinate_dependent": (kinds["coordinate"] + kinds["mixed"]) / max(len(refined), 1),
                         "proposal_success": any(all(s.equivalent(h, th) for s in specs.values()) for h in refined),
                         "proposal_success_L1_only": any(l1.equivalent(h, th) for h in refined),
                         "map_id": hm.hid, "map_description": hm.describe(), "map_kind": classify_atoms(hm),
                         "map_is_spurious": hm.hid.startswith("SPUR"), "map_posterior": float(mpost[imap]),
                         "P_intended_equivalent": float(sum(p for p, e in zip(mpost, eq_all) if e)),
                         "selection_success": eq_all[imap],
                         "L1_heldout_acc": m1["accuracy"], "L1_heldout_f1": m1["violation_f1"],
                         "L1_unseen_aisle_acc": mu["accuracy"], "L3_acc": m3["accuracy"], "L3_f1": m3["violation_f1"],
                         "L3_acc_by_part": parts, "generalization_success": m3["accuracy"] == 1.0})
    write_jsonl(os.path.join(OUT, "runs.jsonl"), rows)
    write_csv(os.path.join(OUT, "runs.csv"), rows)
    write_csv(os.path.join(OUT, "proposals.csv"), prop_rows)
    summ = []
    for cond in ("c1", "c2", "c3"):
        rs = [r for r in rows if r["condition"] == cond and r["status"] == "ok"]
        row = {"condition": cond, "n_runs_ok": len(rs), "n_not_run_or_error": sum(1 for r in rows if r["condition"] == cond and r["status"] != "ok")}
        for m in ("proposal_success", "proposal_success_L1_only", "selection_success", "generalization_success",
                  "map_is_spurious"):
            row[m] = f"{sum(bool(r[m]) for r in rs)}/{len(rs)}"
        for m in ("frac_relational", "frac_coordinate_dependent", "P_intended_equivalent", "L1_heldout_acc", "L1_heldout_f1",
                  "L1_unseen_aisle_acc", "L3_acc", "L3_f1"):
            mu, lo, hi = bootstrap_ci([r[m] for r in rs], 2000) if rs else (float("nan"),) * 3
            row[m] = f"{mu:.2f} [{lo:.2f}, {hi:.2f}]"
            row[m + "_mean"] = mu
        for part in ("vertical", "horizontal", "trap"):
            v = [r["L3_acc_by_part"].get(part) for r in rs if part in r["L3_acc_by_part"]]
            row[f"L3_{part}_acc_mean"] = float(np.mean(v)) if v else None
        summ.append(row)
    write_csv(os.path.join(OUT, "summary.csv"), summ)
    latex(summ)
    for s in summ:
        print(json.dumps(s))


def latex(summ):
    name = {"c1": "Single layout, no guidance", "c2": "Single layout + guidance", "c3": "Two layouts + guidance"}
    L = [r"\begin{table}[t]", r"\centering\small", r"\caption{GPT-5.5 proposals for the aisle norm across store layouts "
         r"(10 independent calls per condition). Proposal: a proposed hypothesis is behaviourally equivalent to the "
         r"intended norm on all three layouts. Selection: the Bayesian MAP hypothesis is. L3 is a test layout never shown "
         r"to the LLM.}", r"\label{tab:layout-generalization}", r"\begin{tabular}{lccccc}", r"\toprule",
         r"Condition & Relational & Proposal & Selection & L1 acc. & L3 acc. / F1 \\", r"\midrule"]
    for s in summ:
        if not s["n_runs_ok"]:
            L.append(f"{name[s['condition']]} & \\multicolumn{{5}}{{c}}{{not run}} \\\\")
            continue
        L.append(f"{name[s['condition']]} & {s['frac_relational_mean']:.2f} & {s['proposal_success']} & "
                 f"{s['selection_success']} & {s['L1_heldout_acc_mean']:.2f} & {s['L3_acc_mean']:.2f} / {s['L3_f1_mean']:.2f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(os.path.join(OUT, "table.tex"), "w") as f:
        f.write("\n".join(L) + "\n")


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("prepare", "run", "analyze"):
        sys.exit("usage: python -m experiments.llm_layout_generalization {prepare,run,analyze}")
    st = sys.argv.pop(1)
    cfg = parse_args("LLM layout generalisation")
    cfg["provider"] = "openai"
    t0 = time.time()
    {"prepare": prepare, "run": run, "analyze": analyze}[st](cfg)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
