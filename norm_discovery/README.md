# Learning the Right Abstraction for Norm Discovery — reconstructed experimental code

This repository rebuilds the experimental setup of *"Learning the Right Abstraction for Norm Discovery"*
(AAMAS 2027 submission) from the paper text, and adds three new experiments:

1. refinement-threshold sensitivity (`experiments/threshold_sensitivity.py`),
2. robustness to noisy demonstrations (`experiments/noise_robustness.py`),
3. identifiability / disambiguating demonstrations (`experiments/identifiability.py`).

No earlier experiment code or data was used. A separate older project exists at `/home/train/norm-abstraction`.
It was neither read nor modified, as instructed.
Results are summarised in **`RESULTS_SUMMARY.md`**. The reconstruction report, with exact trajectories,
hypotheses and sanity checks, is in **`results/reconstruction/RECONSTRUCTION_REPORT.md`**.

## Quick start

```bash
cd /home/train/norm_discovery
PY=/home/train/anaconda3/bin/python          # Python 3.12, numpy 2.3, scipy 1.16, matplotlib 3.10

$PY -m experiments.run_all                   # everything (≈ 15–25 min on 24 cores)
$PY -m experiments.run_all --quick           # smoke test

$PY -m experiments.reconstruct_original      # Phase 1–2: reconstruction report + sanity checks
$PY -m experiments.threshold_sensitivity     # new experiment 1
$PY -m experiments.noise_robustness          # new experiment 2
$PY -m experiments.identifiability           # new experiment 3
$PY -m experiments.supplement_dl_coding      # supplementary: description-length coding sensitivity
$PY -m analysis.aggregate                    # rebuild results/tables/*.md
$PY -m pytest -q tests                       # 23 tests
```

Every runner accepts `--set key=value` (JSON values, e.g. `--set rho=0.2 noise_seeds=10`), `--config file.json`,
`--quick`, and `--out DIR`. All defaults are in `experiments/common.py::DEFAULTS`. Each run writes `metadata.json`
next to its results. That file records the timestamp, the git commit if there is one (otherwise
"unavailable"), a SHA-256 of all source files, library versions, the full configuration and the environment
configuration. All randomness is derived from `base_seed` and the experimental condition
(`experiments.common.seed_for`), so repeated runs give identical results. A test checks this.

## Code layout

```
environment/   gridworld.py (layout, geometric primitives) · layouts.py (Fig. 1 / Fig. 2 stores)
               shopping_domain.py (planning model M, state graph) · trajectories.py (validity, scripts, planner)
               domains.py (per-experiment spec: vocabularies, hidden norm, demos, held-out set, noise generators)
abstractions/  base.py  (formula language, vocabulary families, abstraction = state properties + events)
norms/         hypotheses.py (norm monitors, h = (alpha, N), description lengths) · violations.py (V_h(tau))
inference/     likelihood.py (exact max-ent partition functions, cache, sampler) · posterior.py
               predictive.py (MAP classification, metrics) · refinement.py (Algorithm 1)
proposals/     base.py (provider interface) · deterministic.py (reproducible stand-in for the LLM)
               templates.py (hypothesis JSON templates) · llm_stub.py (optional LLM provider, unused)
baselines/     mlci.py (reconstructed Scobee & Sastry MLCI); fixed / oracle are learner configurations
               (inference/refinement.py::method_config)
experiments/   reconstruct_original.py · threshold_sensitivity.py · noise_robustness.py · identifiability.py
               supplement_dl_coding.py · run_all.py · common.py
analysis/      plotting.py · aggregate.py
tests/         23 tests (validity, violations, events, Z, posterior, predictive, triggering, noise, reproducibility)
results/       raw per-run JSONL/CSV, summaries, zcache/ (partition-function cache)
figures/       all figures (PNG + PDF)
```

## What was reconstructed (and how)

### Planning model (Section III.A)
A deterministic grid. The low-level state is `(pos, cart, mask, hand)`: agent cell; cart at station / held /
parked at a cell; collected required items; hand item (unused by default). Actions are `N S E W PICKUP_CART
LEAVE_CART PICKUP_ITEM(i) EXIT` and every action costs 1. A task context (s0, g) is the start state plus a set of
required items. The goal is EXIT at the exit cell with all items. The `aisle` store also requires the agent to exit
holding the cart (shoppers take the cart to checkout).

* **Exp. 1 store** (`cart`, Fig. 1): a 10×6 grid with one shelf block holding apple/bread/milk/eggs, a cart
  station C next to the entrance E, a return area R at (0,1), and exit X at (9,5).
* **Exp. 2 store** (`aisle`, Fig. 2): a 12×8 grid with three dead-end aisles A1/A2/A3 between shelf blocks
  (items a1/a2/a3 can only be reached from inside an aisle), plus two standalone displays (s1, s2). Aisle A3
  (item a3) is never used in training.

Figures: `figures/gridworld_cart.png`, `figures/gridworld_aisle.png`.

### Abstractions and norms (Section III.B)
* Formulas are JSON: atoms, `not/and/or`, and numeric comparisons. Atoms belong to *vocabulary families*:
  `coords` at(x,y), `cart_possession` hasCart, `cart_location` cartAt(x,y), `hand`, `geometry` (free, adjShelf,
  shelfN/S/E/W, freeWidthH/V), `actions_coarse` (moves, INTERACT, EXIT), and `actions_primitive`
  (PICKUP_CART, LEAVE_CART, PICKUP_ITEM). Each domain defines which families are visible at level 0 (α₀) and
  after refinement (level 1):

  | domain | α₀ (initial) | refinement exposes |
  |---|---|---|
  | cart  | coords, actions_coarse (all object interactions = `INTERACT`) | cart_possession, cart_location, hand, actions_primitive |
  | aisle | coords, cart_possession, actions_coarse | geometry (no aisle predicate exists anywhere) |

* An abstraction α = (α_S, α_A). It contains named **state properties** (formulas over state atoms, which may be
  hierarchical) and **events**, written as a pattern `pre · post` over two consecutive labels. Example (Eq. 12):
  `pickUpCart := (INTERACT ∧ ¬hasCart) · hasCart`. This is Eq. (13) restricted to regular expressions of length
  ≤ 2, which is enough for every abstraction used.
* Norms are compiled to finite monitors:
  * `prohibition(φ)`: one violation for every transition that satisfies φ.
  * `obligation(trigger, discharge)`: once triggered, a later discharge is required before the trajectory ends.
    The obligation can be re-triggered. A discharge on the same transition as the trigger satisfies it.
    `trigger="START"` means the obligation is pending from the start. One violation is counted at EXIT if the
    obligation is still pending.

  V_h(τ) is the number of violations counted by the monitor.

### Description length and prior (Eqs. 18–20)
* L(formula) is the number of nodes: each atom counts 1, each operator 1, and a numeric comparison 3
  (operator, feature, constant).
* L(α) is the sum of the sizes of all definitions. Primitives of α₀ cost 0.
* L(N) = 1 (norm type) plus the sizes of its formulas; `START` costs 1.
* Prior: P(h) ∝ exp(−λL(α) − γL(N)), normalised over the current hypothesis set (`prior_mode=joint`).
  A conditional factorisation P(α)P(N|α) is available as `prior_mode=conditional`.
* **Supplementary coding `tokens`** (`dl_coding=tokens`): identical, except at(x,y) and cartAt(x,y) cost 3
  (the predicate plus 2 integer constants). This makes integer constants count consistently. It was added
  *after* the primary results showed that memorised coordinates are cheap under the node coding. It is reported
  only as a sensitivity analysis.

### Likelihood and partition function (Eqs. 22–23)
P(τ | h, w) = exp(−β(C(τ) + wV_h(τ))) / Z(h, w, s0, g). Z sums over **all** feasible trajectories from s0 that
end in the goal, including arbitrarily long and cyclic ones.

Z is computed **exactly** on the product of the task's reachable state graph (≈ 25–46 k states) with the norm
monitor. The method solves (I − M₀ − e^{−βw}M₁) z = b₀ + e^{−βw}b₁ using one sparse LU factorisation plus a
Woodbury low-rank correction, which gives all 61 values of w at once. M₁ holds the violating transitions.

This was checked against a direct sparse solve, an explicit power series over trajectory lengths, and Monte
Carlo sampling (tests). Results are cached on disk by (hypothesis content, task, β, W).

### Posterior, prediction, refinement (Eqs. 24–32, Algorithm 1)
* Posterior over (h, w_j), with W = {0, 0.5, …, 30} and a uniform P(w|h).
* The predictive p_t = Σ_{h,w} P(τ_t|h,w) P(h,w|D_{1:t−1}) is computed in log space.
* Refinement fires when p_t < ρ·b, with b the running minimum over earlier non-refining demonstrations.
  b is undefined until t = 2, so the first possible trigger is t = 3. The median baseline is a secondary
  diagnostic.
* On refinement, the provider receives D_{1:t} and the next vocabulary level. Its proposals are validated
  (they must be executable over the exposed vocabulary) and added, and the posterior is recomputed on D_{1:t}.
* Held-out classification uses the MAP hypothesis (marginal over w): a trajectory is flagged as violating if
  V ≥ 1.

### Hypothesis proposal (Section IV.A)
`DeterministicProposalProvider` replaces the LLM. It instantiates fixed JSON templates from the demonstrations
it is shown, reading them only through the vocabulary it is allowed to see. All templates are listed in
`results/reconstruction/RECONSTRUCTION_REPORT.md`.

* **Cart, α₀ (from D1).** Coordinate-anchored obligations:
  * `A0_interact_last`: interact at the last-interaction cell before exiting.
  * `A0_visit_last`: visit that cell before exiting.
  * `A0_any_to_last`: after any INTERACT, later interact at that cell.
  * `A0_first_to_last`: after interacting at the first-interaction cell, later interact at the last one.
  * `H_null`: no norm.
* **Cart, level 1.** `H_int` (intended: pickUpCart → returnCart), `H_uncond` (returnCart unconditionally),
  `H_item` (pickUpItem → returnCart, the competitor discussed in the paper), `H_visit`, `H_leave` (return
  anywhere), and `H_noexit` (no EXIT while holding the cart). The return cell is read from where demonstrators
  performed LEAVE_CART.
* **Aisle, α₀ (from D1).** The cart is forbidden in the cells the demonstrator visited without it
  (`B0_cells`), at the pick cells (`B0_pickcells`), or when interacting there (`B0_interact_cells`), plus `H_null`.
* **Aisle, level 1.**
  * `S_inAisle` (intended): inAisle := (shelfW ∧ shelfE) ∨ (shelfN ∧ shelfS); FORBIDDEN hasCart ∧ inAisle.
  * `S_corridor`: freeWidth ≤ 1. It has the same description length and agrees with the intended rule on every
    held-out case.
  * `S_nearShelf`: refuted by D3.
  * `S_noShelfInteractWithCart`.
  * `S_memorised`: a disjunction of every cell demonstrators visited without the cart, i.e. coordinate
    memorisation re-proposed at refinement.

The `LLMProposalProvider` stub builds a prompt and parses JSON hypotheses with the same schema. It is not used in
any reported number.

### Baselines
* **Fixed abstraction**: the same learner, with refinement disabled.
* **Oracle abstraction**: the level-1 vocabulary and its hypotheses (instantiated from D1) are available from
  t = 1 together with the α₀ hypotheses. There is no refinement.
* **MLCI** (`baselines/mlci.py`), after Scobee & Sastry:
  * Demonstrations come from a max-ent distribution restricted to trajectories that satisfy *hard*
    constraints.
  * Candidate constraints forbid transitions satisfying A ∧ B, with A ∈ {⊤, at(x,y), cartAt(x,y)} and
    B ∈ {⊤, hasCart, ¬hasCart, N, S, E, W, PICKUP_CART, LEAVE_CART, PICKUP_ITEM, INTERACT, EXIT}. MLCI gets the
    full low-level state, including primitive actions, but no invented predicates and (by default) no geometric
    relations. `mlci_geometry=true` is an ablation.
  * Greedy loop:
    1. Compute the exact expected visitation of each candidate under the current constrained max-ent model.
    2. Exclude candidates violated by any demonstration.
    3. Compute the exact log-likelihood gain of the top 5.
    4. Add the best one.
    5. Stop when the gain falls below ε, or at 30 constraints.
  * ε = log(#admissible candidates) ≈ 7 nats, fixed *a priori* as a Bonferroni/BIC-style correction. It is the
    MLCI analogue of the description-length prior. ε ∈ {0.1, 1, 15} is reported only as a sensitivity check.
  * Prediction: a trajectory is violating if it contains a forbidden transition.

## Assumptions and interpretation choices (all configurable unless stated)

1. **Store mechanics.**
   * All actions cost 1.
   * An item can only be picked if the cart is held or parked within Manhattan distance 2 (`cart_reach`).
     Without this rule the optimal shopper returns the cart early and carries the last item by hand, which
     contradicts the paper's demonstrations. Hand carrying is available as `hand_capacity=1`.
   * Exp. 2 requires exiting with the cart (`exit_requires_cart`).
   * Items count as collected once picked.
2. **Layouts** follow Fig. 1 and Fig. 2 approximately. The exit is X; Fig. 1's caption says "exits at E".
3. **α₀ for Exp. 1** contains raw coordinates and coarse actions, following Section III.B's description of α₀.
   This makes coordinate-anchored obligations expressible at α₀. That matters for the results (see the summary).
4. **Events** are patterns of length ≤ 2. Event and norm formulas are evaluated on transitions. The terminal
   label is the state after EXIT.
5. **Obligation semantics**: re-triggerable; a discharge in the same step satisfies it; one violation per
   pending obligation at the end. Prohibitions count one violation per step.
6. **Description-length coding** is node count (primary). The `tokens` coding is supplementary (see above).
7. **Prior normalisation** is joint over the current hypothesis set.
8. **Hypothesis proposals** come from deterministic templates. Every template set contains the intended
   hypothesis and competitors, including coordinate-memorising and differently-triggered norms. Level-0
   templates use D1 only (Alg. 1, line 1). A null hypothesis is always included.
9. **Oracle condition** = α₀ hypotheses plus level-1 hypotheses instantiated from D1, available at t = 1.
10. **Training demonstrations** are shortest trajectories under the hidden norm with fixed tie-breaking
    (N, S, E, W, …).
    * Exp. 1 tasks: {apple, milk}, {bread, eggs}, {apple, bread, milk}. All three need a cart, so each
      demonstration acquires, uses and returns one; at least one has two items, as the paper requires.
    * Exp. 2 tasks: {a1}, {a2}, {s1}.

    A supplementary run uses the same tasks with random tie-breaking (20 seeds).
11. **Held-out sets** are scripted, and their labels are verified against the hidden norm.
    * Exp. 1: 10 trajectories. 4 are compliant with different items and orders. 6 are violations: 2 exit
      without returning the cart, 2 leave it at a wrong location, and 2 return it and then reacquire it.
    * Exp. 2: 10 trajectories (5 compliant, 5 violating). The 5 that involve the unseen aisle A3 make up the
      "unseen" subset.
12. **Pools for the new experiments.**
    * Cart tasks are item subsets of size 1–3. Aisle tasks are subsets of {a1, a2, s1, s2} of size 1–2; A3 is
      excluded so that the unseen test stays meaningful.
    * Demonstrations are optimal with random tie-breaking.
    * Noise corrupts round(q·n) randomly chosen positions. Clean demonstrations are paired across conditions,
      and the corrupted sets are nested in q.
13. **Behavioural noise** inserts 1–3 random out-and-back detours (2–6 steps each). With probability 0.5 it also
    inserts a redundant LEAVE_CART/PICKUP_CART pair. The result is checked to remain compliant.
14. **Violation noise** produces valid trajectories that violate the hidden norm. Cart: no return, wrong
    location, or reacquire (uniformly). Aisle: the cart is taken straight into the aisle.
15. **Identifiability contexts** vary the required items (0–2) and the initial cart location (station or
    parked at any free cell). The agent always starts at E.

## Outputs

See `RESULTS_SUMMARY.md` for the full list of raw-result files and figures.
