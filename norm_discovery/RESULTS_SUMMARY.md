# Results summary

This file reports two kinds of experiments. They are kept strictly separate.

* **Part I – actual GPT-5.5 proposal experiments.** Hypotheses are proposed and refined by `gpt-5.5` through the
  OpenAI API. Every call returned model `gpt-5.5-2026-04-23`. Results: `results/gpt55/`, figures:
  `figures/gpt55/`.
* **Part II – deterministic-proposal ablation (inference only, NOT GPT-5.5).** An LLM-free template provider
  isolates the Bayesian inference and refinement machinery. Results: `results/deterministic_proposals/`,
  figures: `figures/deterministic_proposals/`.

Brackets give 95% bootstrap CIs. ± gives the SD over runs. "Full" means the paper's method with ρ = 0.1 and the
running-minimum baseline, unless stated otherwise.

---

# Part I – Actual GPT-5.5 proposal experiments

**Status.**
* Done: smoke test, original reproduction (10 runs per domain), threshold sensitivity (10 runs per ρ and domain),
  and identifiability (cart 10/10 runs).
* Stopped: **the OpenAI account ran out of credits at 11:46** (HTTP 429 `insufficient_quota` /
  `credit_balance_exhausted`). Identifiability for the aisle domain has only 4/10 runs; 6 failed with that error,
  which is recorded in their `metrics.json`. **The GPT-5.5 noise experiment was not completed:** only about 10
  cart full-method runs (out of 180 GPT runs) finished, so no noise results are reported for GPT-5.5. Per Section
  12 of the instructions, nothing fell back to another provider or model.
* Usage: 103 successful API calls, 264k input and 442k output tokens. All responses are cached in
  `results/llm_cache/`, so re-running after topping up reuses them at no cost.

Generation settings: the API rejects `temperature` for `gpt-5.5` (HTTP 400 in the first smoke test), so the
paper's temperature of 0.2 could not be used. Temperature is unset (API default). Reasoning effort is the API
default. Each call requests 8 hypotheses. Model identity was checked on every call.

## I.1 Original protocol, 10 independent GPT-5.5 runs per domain (`results/gpt55/original/`)

| Method | Exp.1 Acc. | Exp.1 Viol. F1 | Exp.2 Acc. | Exp.2 Viol. F1 | Exp.2 Unseen Acc. |
|---|---|---|---|---|---|
| Fixed abstraction (GPT-5.5 initial proposals) | 0.80 ± 0.00 | 0.80 ± 0.00 | 0.60 ± 0.00 | 0.33 ± 0.00 | 0.40 ± 0.00 |
| MLCI (no LLM) | 0.60 | 0.75 | 0.50 | 0.00 | 0.40 |
| **Ours – GPT-5.5** | **0.80 ± 0.00** | **0.80 ± 0.00** | **0.60 ± 0.00** | **0.33 ± 0.00** | **0.40 ± 0.00** |
| Oracle abstraction (GPT-5.5 proposals over the refined vocabulary from t = 1) | 0.96 ± 0.08 | 0.96 ± 0.08 | 0.98 ± 0.06 | 0.98 ± 0.05 | 1.00 ± 0.00 |
| *Paper, Ours – GPT-5.5* | *0.80 ± 0.00* | *0.80 ± 0.00* | *0.96 ± 0.08* | *1.00 ± 0.00* | *0.93 ± 0.14* |

| Domain | Method | Refined | Intended proposed | Final MAP = intended | Mean P(intended) |
|---|---|---|---|---|---|
| cart | GPT-5.5 full | 0/10 | 0/10 | 0/10 | 0.00 |
| cart | GPT-5.5 oracle | – | 7/10 | 5/10 | 0.56 |
| aisle | GPT-5.5 full | 0/10 | 0/10 | 0/10 | 0.00 |
| aisle | GPT-5.5 oracle | – | 10/10 | 7/10 | 0.47 |

"Intended" means behaviourally equivalent to the hidden norm on a fixed probe set of 541 (cart) / 372 (aisle)
trajectories. GPT phrases norms in its own syntax, so exact template matching would undercount. No proposals were
rejected, except one level-1 aisle proposal in one run.

**Refinement trigger with GPT-5.5 hypotheses.** The deterministic result carries over, and the GPT-5.5 margin is
much larger. In all 20 runs, refinement did not trigger at ρ = 0.1:
* Cart: log p₂ ≈ −26.9 and log p₃ ≈ −23.4, so p₃/b₃ ≈ 36.
* Aisle: log p₂ ≈ −14.6 and log p₃ ≈ −5.0, so p₃/b₃ ≈ 1.4 × 10⁴.

The reason is that GPT's initial proposals explain the demonstrations about equally well, or badly, at t = 2 and
t = 3:
* Cart: the MAP in 10/10 runs is "must INTERACT at (0,1) before exiting", an unconditional coordinate obligation.
  It is a worse explanation than my template competitor (log p₂ −26.9 vs −11.8), but p₃ is not lower than p₂.
* Aisle: coordinate rules such as "no cart at (3,1)/(3,2)" or "no N from (3,3) with the cart".

Full traces (p_t, b_t, ρ·b_t, p_t/b_t, trigger, P(h|D) for every hypothesis) are in each run's `trace.json`. The
paper's GPT-5.5 Exp. 2 numbers (0.96 / 1.00 / 0.93) are not reproduced. Exp. 1 matches numerically (0.80), but for
a different reason: the paper attributes its 0.80 to an item-triggered competitor selected *after* refinement,
whereas here refinement never happens.

When GPT-5.5 is given the refined vocabulary directly (oracle), its proposals are good:
* Aisle: "no cart between opposing shelves" (shelfE ∧ shelfW), in 10/10 runs.
* Cart: "after taking a cart it must be returned to the bay", in 7/10 runs.

The bottleneck is therefore the trigger, not GPT-5.5's ability to propose abstractions.

## I.2 Refinement-threshold sensitivity with GPT-5.5 (`results/gpt55/threshold_sensitivity/`, `figures/gpt55/threshold_gpt55.png`)

Original 3-demonstration protocol; ρ ∈ {0.01, …, 0.8}; 10 GPT-5.5 runs per ρ and domain (160 runs).
* **No run refined at any ρ.** Accuracy is 0.80 (cart) and 0.60 (aisle, unseen 0.40) at every ρ, with
  refinement rate 0 and P(intended) 0.
* This follows from I.1. Every run has p₃/b₃ > 1, so the trigger p₃ < ρ·b₃ cannot fire for any ρ < 1. On this
  protocol the outcome is invariant to ρ.
* No new API calls were needed: every prompt equals the same run's prompt from I.1 and was reused from that run's
  cache.

## I.3 Noise robustness with GPT-5.5 — **not completed** (`results/gpt55/noise_robustness/`)

The API credits ran out about 15 minutes into the sweep. The per-run directories that exist are partial: mostly
the cart q = 0 condition, with 13 recorded credit errors. They are **not** analysed or reported. The dry-run
estimate (`results/gpt55/estimate/estimate.json`) for the full sweep with 10 GPT runs per condition is at most
656 calls, about 1.9M input and 2.4M output tokens. Re-run with `python -m experiments.gpt55 noise` after topping
up; cached responses are reused.

## I.4 Identifiability with GPT-5.5 (`results/gpt55/identifiability/`)

**A. Proposal quality.** Per run: initial proposals from D1, then refinement proposals from D1–D3, with refinement
forced so that proposal quality is measured independently of the trigger. The table counts runs in which some
GPT-5.5 hypothesis is behaviourally equivalent to the reference hypothesis.

| Domain | Reference hypothesis | Proposed in runs |
|---|---|---|
| cart | **H_int (intended)** | **7/10** |
| cart | A0_interact_last (coordinate: "interact at (0,1) before exit") | 10/10 |
| cart | H_noexit ("no exit while holding the cart") | 7/10 |
| cart | A0_first_to_last (coordinate C→R obligation) | 2/10 |
| cart | A0_any_to_last, A0_visit_last, H_item (paper's competitor), H_uncond, H_visit, H_leave | 0/10 |
| aisle (4 runs; 6 failed on credits) | **S_inAisle (intended)** | **2/4** |
| aisle | S_corridor (width-1 corridor) | 2/4 |
| aisle | B0_cells / B0_pickcells (memorised cells) | 3/4 / 2/4 |
| aisle | S_nearShelf, S_memorised, S_noShelfInteractWithCart | 0/4 |

GPT-5.5 never proposed the paper's item-triggered competitor, nor the coordinate rule that dominated the
deterministic ablation (`A0_any_to_last`).

**B. Bayesian identifiability given each run's own GPT-5.5 hypothesis set** (cart; no extra API calls). Each set
was combined with the diagnostic, control and reacquire sequences from Part II.

| Sequence | P(intended) after D1 | after D5 | after D6 | after D8 | Mean accuracy at D8 |
|---|---|---|---|---|---|
| control (ordinary only) | 0.635 | 0.635 | 0.635 | 0.635 | 0.96 |
| diagnostic (D6 = parked-cart context) | 0.635 | 0.635 | **0.700** | 0.700 | 0.96 |
| reacquire (D6 = return–reacquire–return) | 0.635 | 0.635 | 0.635 | 0.635 | 0.96 |

0.70 is the ceiling, because the intended norm is in the set in 7/10 runs.
* In 6 of those 7 runs, the intended hypothesis already has P ≈ 1.00 after the first demonstration. GPT's
  competitors (mostly unconditional "interact at (0,1)" rules) allow cheap loopholes and lose by the size
  principle.
* In the one ambiguous run (run 6, competitor "FORBIDDEN: EXIT ∧ ¬cart parked in return bay"), P(intended) is 0.35
  after D5 and 1.00 after the diagnostic demonstration.
* As in Part II, the reacquire demonstration changes nothing.

**Proposal performance vs. statistical identifiability.**
* With GPT-5.5's candidate sets, the identifiability problem of Part II is mostly absent. GPT rarely proposes the
  near-equivalent coordinate competitor.
* The limiting factor is whether GPT proposes the intended norm at all: 7/10 cart runs, 2/4 aisle runs.
* The diagnostic-context mechanism still works when real ambiguity occurs (run 6).

---

# Part II – Deterministic-proposal ablation (inference only; NOT GPT-5.5)

The paper's LLM is replaced by a deterministic template provider (see README). These results describe the
*inference and refinement machinery* given a fixed, known set of proposals. They say nothing about LLM proposal
quality.

## 0. Reconstruction and sanity checks (`results/deterministic_proposals/reconstruction/`)

### Original protocol: 3 demonstrations, canonical order

| Domain | Method | Acc. | Viol. F1 | Unseen acc. | MAP hypothesis | P(intended) | Refined? |
|---|---|---|---|---|---|---|---|
| cart  | fixed  | 0.80 | 0.80 | – | A0_any_to_last | – | – |
| cart  | full   | 0.80 | 0.80 | – | A0_any_to_last | – (not proposed) | **no** |
| cart  | oracle | 0.80 | 0.80 | – | A0_any_to_last | 0.152 | – |
| cart  | MLCI   | 0.60 | 0.75 | – | 6 constraints | – | – |
| aisle | fixed  | 0.60 | 0.33 | 0.40 | B0_cells | – | – |
| aisle | full   | 0.60 | 0.33 | 0.40 | B0_cells | – (not proposed) | **no** |
| aisle | oracle | 1.00 | 1.00 | 1.00 | S_corridor (≡ S_inAisle on all held-out) | 0.382 | – |
| aisle | MLCI   | 0.50 | 0.00 | 0.40 | 0 constraints | – | – |

The paper reports full method / GPT-5.5: 0.80, 0.80 for Exp. 1 and 0.96, 1.00, 0.93 for Exp. 2. Fixed
abstraction: 0.60, 0.50 and 0.20, 0.33, 0.33. MLCI: 0.60, 0.50 and 0.60, 0.50, 0.33. Oracle: 1.0 everywhere.

**The reconstruction does not reproduce the paper's headline gain on the original protocol.** With ρ = 0.1,
refinement is never triggered in either domain, so the full method equals the fixed abstraction.

* Under Algorithm 1 the baseline b is undefined at t = 2, so the *only* possible trigger is at t = 3, by
  comparing p₃ with p₂.
* **Cart:** p₃/p₂ = e^−0.44 ≈ 0.64. The α₀ hypothesis `A0_any_to_last` ("after any INTERACT, eventually
  INTERACT at (0,1)") explains all three demonstrations about as well as the intended norm. It is a
  coordinate-anchored version of the norm, expressible because α₀ keeps raw coordinates (README assumption 3).
  It misclassifies only the two return-then-reacquire violations, which gives 0.80.
* **Aisle:** p₃/p₂ = e^+9.5. D3 (standalone display) is a short trajectory, so its predictive probability is
  high regardless of whether the norm is adequate.

With random tie-breaking of the same demonstration tasks (20 seeds), every Bayesian condition gives the same
numbers. MLCI drops to 0.39 [0.36, 0.42] in the cart domain, and the aisle oracle gets 0.95 [0.88, 1.00]
(`randomized_original_summary.csv`).

### Sanity checks (`sanity_checks.json`)

* **Exact Z.** The fixed-point/Woodbury partition functions equal a direct sparse solve to machine precision.
  Tests also check them against an explicit power series and Monte Carlo sampling.
* **The intended hypothesis is right and identifiable from ordinary demos.** It has zero violations on training
  and held-out accuracy 1.0 in both domains. Aisle: the oracle recovers the structural norm (MAP `S_corridor`,
  a tied, held-out-equivalent variant; P(`S_inAisle`) = 0.38). Cart: with only 3 demonstrations the oracle
  prefers the shorter coordinate rule (P = 0.75 vs 0.15 for H_int). With 20 ordinary demonstrations the oracle
  selects H_int in 20/20 seeds (accuracy 1.00).
* **α₀ cannot solve the abstraction-dependent cases.**
  * Cart: of 12,430 enumerated α₀ norms, 9,257 are consistent with the training data. *None* classifies both
    the compliant and the reacquire cases correctly. The best (selected on held-out) reaches 0.80.
  * Aisle: among α₀ rules that name only cells seen in training, the best unseen accuracy is 0.40. A rule naming
    an A3 cell reaches 1.0 on unseen cases only when chosen post hoc with held-out knowledge.
* **The unseen-aisle test separates memorisation from structure.** Memorised A1+A2 cells give 1.00 on seen
  held-out cases and 0.40 on unseen ones. The structural predicate gives 1.00 on unseen ones.

---

## 1. Refinement-threshold sensitivity (`results/deterministic_proposals/threshold_sensitivity/`)

**Protocol.**
* Full method with ρ ∈ {0.01, 0.025, 0.05, **0.1**, 0.2, 0.4, 0.6, 0.8}, using the running-minimum baseline
  (primary) and the median baseline (secondary).
* (A) The original 3 demonstrations in all 6 orders.
* (B) 20 independent clean sequences of 20 demonstrations each (random pool tasks, optimal behaviour, random
  tie-breaking).
* Fixed and oracle runs on the same sequences serve as references.
* Metrics: accuracy, violation F1, unseen accuracy, whether refinement happened, the number of refinements and
  the demonstration index at which they happened, the number of hypotheses, and P(intended). Every p_t, b_t and
  p_t/b_t is in `traces.csv`.

**Pool results, accuracy (refinement rate):**

| ρ | cart, running min | cart, median | aisle, running min | aisle, median |
|---|---|---|---|---|
| 0.01 | 0.80 (0.00) | 0.80 (0.00) | 0.61 (0.35) | 0.69 (0.75) |
| 0.025 | 0.81 (0.05) | 0.81 (0.05) | 0.62 (0.40) | 0.69 (0.75) |
| 0.05 | 0.81 (0.05) | 0.81 (0.05) | 0.62 (0.45) | 0.70 (0.85) |
| **0.1** | **0.81 (0.05)** | 0.84 (0.20) | **0.63 (0.50)** | 0.70 (0.95) |
| 0.2 | 0.84 (0.20) | 0.91 (0.55) | 0.68 (0.80) | 0.70 (1.00) |
| 0.4 | 0.85 (0.25) | 0.99 (0.95) | 0.69 (0.85) | 0.70 (1.00) |
| 0.6 | 0.92 (0.60) | 0.99 (0.95) | 0.69 (0.95) | 0.70 (1.00) |
| 0.8 | 0.94 (0.70) | 1.00 (1.00) | 0.69 (0.95) | 0.70 (1.00) |
| reference | fixed 0.80, oracle 1.00 | | fixed 0.56, oracle 0.99 (unseen 0.97) | |

Mean P(intended) for the cart domain rises from 0.03 (ρ = 0.1, min) to 0.47 (ρ = 0.8, min) and to 0.66
(median, ρ ≥ 0.4).

**Original protocol:**
* Cart: no refinement in any order for ρ ≤ 0.4; refinement in 1/6 orders at ρ = 0.6 and 2/6 at ρ = 0.8.
  Accuracy stays at 0.80 throughout.
* Aisle: refinement in 2/6 orders at every ρ, rising to 3/6 at ρ ≥ 0.6. Accuracy is 0.60–0.63 because the
  refined MAP is a memorising or α₀ hypothesis.

**Weakness of the running-minimum baseline** (`running_min_diagnostics.json`, `threshold_logp_vs_length.png`).
* log p_t is driven mostly by demonstration length (task cost), not by how well the norm explains the
  demonstration. Spearman ρ(log p_t, length) = −0.94 (cart) and −0.71 (aisle). Length alone explains R² = 0.93
  and 0.83 of the variance in log p_t. Each extra step costs 0.36 and 0.61 nats respectively.
* Because b_t is a running *minimum*, a single long, hard demonstration lowers the threshold for the rest of the
  sequence. The trigger then becomes progressively harder to reach (`threshold_traces.png`).
* The median baseline does not shrink this way. It triggers more often and gives higher accuracy at every ρ
  (cart: 0.84 vs 0.81 at ρ = 0.1, and 1.00 vs 0.94 at ρ = 0.8).

**Does the conclusion depend on ρ = 0.1?** Yes.
* In the cart domain the benefit of abstraction refinement is essentially absent at ρ ≤ 0.1 (0.80–0.81, about
  the fixed abstraction's level). It appears only at ρ ≥ 0.4–0.6 or with the median baseline, where it reaches
  the oracle's 1.00.
* In the aisle domain the result is nearly insensitive to ρ above 0.2, but for a different reason. Refinement
  does happen (up to 95–100% of runs), yet the selected hypothesis is coordinate memorisation (`S_memorised`).
  Unseen-aisle accuracy stays at 0.40 for every ρ. This is a hypothesis-*selection* problem (the
  description-length prior), not a triggering problem (see the Supplement).

---

## 2. Robustness to noisy demonstrations (`results/deterministic_proposals/noise_robustness/`)

**Protocol.**
* 2 domains × 2 noise types × q ∈ {0, 5, 10, 20, 30}% × 20 seeds × 4 methods = 1,440 runs.
* Each run uses 20 training demonstrations from the task pool, with round(q·20) of them corrupted.
* Clean demonstrations are paired across conditions, corrupted sets are nested in q, and the held-out set is
  fixed and clean.
* Per-seed raw results, including every training trajectory, are in `runs.jsonl`.

**Behavioural (suboptimality) noise. The label is unchanged.**

| q | cart full | cart fixed | cart oracle | cart MLCI | aisle full | aisle fixed | aisle oracle | aisle MLCI |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.82 [0.80,0.85] | 0.80 | 1.00 | 0.79 [0.72,0.86] | 0.62 [0.58,0.65] | 0.58 | 0.92 [0.84,0.98] | 0.59 |
| 0.05 | 0.99 [0.97,1.00] | 0.80 | 1.00 | 0.80 | 0.70 | 0.58 | 0.97 | 0.69 |
| 0.10 | 0.99 | 0.80 | 1.00 | 0.82 | 0.70 | 0.58 | 0.97 | 0.69 |
| 0.20 | 1.00 | 0.80 | 1.00 | 0.83 | 0.68 | 0.58 | 0.97 | 0.69 |
| 0.30 | 1.00 | 0.80 | 1.00 | 0.82 | 0.67 | 0.58 | 0.97 | 0.69 |

Inference itself is robust to behavioural noise. Detours are compliant, so they leave the likelihood ratios
between hypotheses unchanged, and oracle accuracy is flat. The full method in the cart domain *improves* sharply
(0.82 → 0.99 at a single noisy demonstration; mean refinement calls 0.1 → 4.9). The reason is the trigger, not
better inference: inefficient demonstrations are long, long demonstrations have low p_t, and that trips the
running-minimum threshold. Refinement is therefore set off by inefficiency rather than by norm inadequacy. This
is the confound documented in Section 1. Aisle unseen accuracy stays at 0.40 for full, fixed and MLCI, and is
0.90–0.94 for the oracle.

**Norm-violation noise.**

| q | cart full | cart fixed | cart oracle | cart MLCI | aisle full | aisle fixed | aisle oracle | aisle MLCI |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.82 | 0.80 | 1.00 | 0.79 | 0.62 | 0.58 | 0.92 | 0.59 |
| 0.05 | 0.84 [0.81,0.88] | 0.80 | 0.92 [0.88,0.96] | 0.74 [0.66,0.80] | 0.58 | 0.57 | 0.65 [0.56,0.73] | 0.47 |
| 0.10 | 0.82 | 0.80 | 0.87 [0.83,0.91] | 0.62 [0.54,0.70] | 0.57 | 0.57 | 0.56 | 0.43 |
| 0.20 | 0.78 | 0.78 | 0.82 | 0.49 [0.44,0.57] | 0.56 | 0.56 | 0.55 | 0.41 |
| 0.30 | 0.75 [0.69,0.80] | 0.75 | 0.77 [0.72,0.80] | 0.42 [0.40,0.47] | 0.56 | 0.55 | 0.55 | 0.40 |

Violation F1 (cart, q = 0 → 0.30): oracle 1.00 → 0.75, full 0.82 → 0.71, MLCI 0.74 → 0.05. Aisle oracle unseen
accuracy: 0.90 → 0.63.

* **Even one violating demonstration in 20 hurts the Bayesian learner.** With the oracle vocabulary, the cart
  MAP is the intended norm in 20/20 seeds at q = 0, in 12/20 at 5%, and in 2/20 at 20%. At higher q the
  posterior moves to `A0_any_to_last`, which some violation types (re-acquisition) do not violate.
* **In the aisle domain the oracle collapses to `S_nearShelf`.** It is inferred with a small violation cost w,
  but MAP classification ignores w and flags every compliant trajectory that passes a shelf. Accuracy falls to
  ≈ 0.55.
* **MLCI degrades the most.** Hard constraints cannot be violated by any demonstration, so a single violating
  demonstration removes the true constraint from consideration. The remaining spurious state constraints flag
  compliant held-out trajectories.
* We do not observe the expected advantage of the full method over the fixed abstraction under violation noise;
  the two coincide for q ≥ 0.2.

---

## 3. Identifiability and disambiguating demonstrations (`results/deterministic_proposals/identifiability/`)

**Protocol.**
* Cart domain, using the oracle hypothesis set (11 hypotheses, held fixed) so that identifiability is studied
  independently of the trigger.
* All demonstrations are optimal under the hidden norm.
* Candidate demonstration contexts: 0–2 required items × initial cart location (station, or parked at any of
  the 55 free cells), giving 616 contexts.
* Because every cart hypothesis has binary V, the likelihood ratio of two hypotheses depends on a demonstration
  only through (V₁, V₂). The exact expected information gain I(H; τ | context) is therefore computed from four
  path masses, with no sampling. It agrees with the main Z solver to machine precision.
* No probabilities are set by hand; all ambiguity comes from the likelihoods.

**Ambiguity after ordinary demonstrations.** After D1–D5 (ordinary tasks): P(H_int) = 0.198,
P(`A0_any_to_last`) = 0.751, P(`H_item`) = 0.051. The two leading hypotheses agree on every ordinary
demonstration; both require the return trip. Each ordinary demonstration adds only +0.134 nats of evidence for
H_int. This comes from the size principle: H_int also rules out "return, then reacquire and leave". The prior
favours the shorter coordinate rule by 2.0 nats.

**Diagnostic demonstration.**
* The search, using only the posterior (no knowledge of the truth), selects the context "bread, cart already
  parked at (3,1)". Its MI is 0.512 nats, which equals the entropy of the two-hypothesis posterior, so one
  demonstration there is expected to resolve the pair completely.
* The true demonstrator puts the bread into the parked cart without performing a pick-up and leaves without
  returning it. This is compliant under "returning is triggered by *taking* a cart" but violates "every
  interaction must be followed by an interaction at R".
* After this single demonstration, P(H_int) goes from 0.198 to 1.000 (log Bayes factor int:alt from 0.67 to
  18.9), and held-out accuracy goes from 0.80 to 1.00 (`identifiability_posterior.png`).

**Controls.**
* *Redundant demonstrations.* With D4–D8 ordinary, P(H_int) moves from 0.198 to only 0.277 at t = 8. In the
  long control with 20 random ordinary demonstrations (10 seeds), P(H_int) crosses 0.5 only at t = 15 and
  reaches 0.75 at t = 23. Accuracy stays at 0.80 until the crossing.
* *The intuitively diagnostic "return, reacquire, return again" demonstration.* It changes the posterior by
  exactly the same amount as an ordinary demonstration (+0.134 nats). Under the trajectory-level max-ent model,
  a demonstration compliant with both hypotheses carries evidence only through its task context, never through
  which compliant path was taken.
* *One extra demonstration after D1–D5:*

  | Extra demonstration | P(H_int) after it | MAP = H_int |
  |---|---|---|
  | actively selected | 1.000 | 1/1 |
  | random context (50 seeds) | 0.32 [0.26, 0.39] | 7/50 |
  | random ordinary context (50 seeds) | 0.223 [0.223, 0.223] | 0/50 |

* *MI landscape (equal pair prior).* Ordinary contexts already separate most competitors almost perfectly from
  H_int: MI ≈ log 2 for H_uncond, H_visit, H_leave, H_noexit and the A0 interact/visit/first-to-last rules.
  The exceptions are exactly the coordinate rule `A0_any_to_last` (best ordinary-context MI 0.046) and the
  paper's item-triggered competitor `H_item` (0.14). These need a non-standard context, a pre-parked cart, to be
  told apart (MI → log 2).

**Answer to the scientific question.** Two normative abstractions are observationally indistinguishable when
the task contexts that are demonstrated never make their compliant behaviours diverge. More demonstrations of
the same kind barely help. What separates them is a demonstration in a context where one hypothesis requires
behaviour that the other forbids or makes unnecessary. Such contexts can be found automatically from the
posterior.

---

## Supplement: description-length coding (`results/deterministic_proposals/supplement_dl_coding/`)

Added *after* the primary results, as a sensitivity analysis. The primary coding charges 1 unit for at(x,y).
The token-consistent coding charges 3 (predicate + x + y), matching how numeric constants are already counted
in comparisons. Same datasets as Experiment 1 (pool, 20 seeds).

| Aisle domain, pool | nodes (primary): acc / unseen | tokens: acc / unseen |
|---|---|---|
| full ρ = 0.1, min | 0.63 / 0.40 | 0.78 / 0.70 |
| full ρ = 0.4, min | 0.69 / 0.40 | 0.94 / 0.91 |
| full ρ = 0.8, median | 0.70 / 0.40 | 1.00 / 1.00 |
| oracle | 0.99 / 0.97 | 1.00 / 1.00 |

Cart results are unchanged. **Whether the structural aisle abstraction beats coordinate memorisation of the two
training aisles depends entirely on how coordinate literals are priced in L(·).** Under the primary coding,
memorising 4 cells (L = 8) is cheaper than defining inAisle (L = 11). The paper does not specify L, so this is a
substantive modelling choice to report.

---

## Surprising findings and limitations

* **Headline results not reproduced.** With the paper's ρ = 0.1, refinement never fires on the original
  3-demonstration protocol. Algorithm 1's trigger can only fire at t = 3, and p_t mostly reflects trajectory
  length.
* **Weak competitors versus near-equivalent ones.** The cart domain's real competitor is a coordinate-anchored
  α₀ obligation that is cheaper under the description-length prior. The paper's item-triggered competitor is
  never the MAP here.
* **MAP classification ignores w.** A hypothesis selected with a near-zero violation cost still classifies by
  V ≥ 1. This drives the aisle collapse under violation noise.
* **Proposals are deterministic templates, not an LLM.** Results reflect inference given those proposals. In
  particular, the memorisation competitor `S_memorised` is re-proposed at refinement. An LLM might not propose
  it; our results show what happens if it does.
* **Small worlds.** Small deterministic grids; one norm per domain; 10 held-out trajectories per domain (so
  accuracy moves in steps of 0.1); demonstrations are optimal or explicitly perturbed.
* **Reconstruction choices.** Layouts, mechanics (`cart_reach`, `exit_requires_cart`), event length ≤ 2, the
  description-length coding and the MLCI stopping threshold are our reconstruction choices (README "Assumptions").
  MLCI is our reimplementation, not the original code.

---

## Proposed paragraphs for the paper (deterministic-proposal ablation; see the GPT-5.5 section for what changes)

**Refinement-threshold sensitivity.** We varied the refinement threshold ρ from 0.01 to 0.8 on the original
three demonstrations (all six orders) and on 20 independently generated sequences of 20 demonstrations. On the
original protocol, refinement was never triggered at ρ = 0.1 in either domain. Because the baseline is undefined
for the second demonstration, the trigger can only fire at the third. On the longer sequences, abstraction
refinement improved held-out accuracy in the cart domain only for larger thresholds: 0.81 at ρ = 0.1 versus
0.94 at ρ = 0.8 with the running-minimum baseline, and up to 1.00 with a median baseline (fixed abstraction 0.80,
oracle 1.00). The predictive probabilities that drive the trigger are dominated by demonstration length:
Spearman correlation −0.94 and −0.71 in the two domains, with length alone explaining 93% and 83% of the
variance. A running-minimum baseline therefore raises the bar after any long demonstration. Refinement decisions
should consequently be based on a length-normalised or hypothesis-relative statistic rather than on raw
predictive probabilities with a hand-picked ρ.

**Robustness to noisy demonstrations.** We trained on 20 demonstrations of which 0–30% were corrupted, over 20
seeds per condition, and kept the held-out set fixed. Behavioural noise (compliant detours) did not affect
inference with the oracle abstraction: accuracy was 1.00 (cart) and 0.97 (aisle). In the cart domain it even
made the full method reach 0.99–1.00, because long noisy demonstrations trip the refinement trigger. Genuine
norm violations were more damaging. With the oracle abstraction, accuracy in the cart domain fell from 1.00 to
0.92 with a single violating demonstration and to 0.77 at 30%, as the posterior shifted to a coordinate-anchored
alternative. In the aisle domain it fell from 0.92 to 0.55, because a weakly weighted "no cart near any shelf"
hypothesis became MAP. The hard-constraint MLCI baseline degraded most (cart 0.79 → 0.42, violation F1
0.74 → 0.05), since any violating demonstration excludes the true constraint. Explicitly modelling demonstrator
noise, and accounting for the inferred violation cost when classifying, are necessary before these methods are
applied to imperfect human behaviour.

**Identifiability and diagnostic demonstrations.** With the correct vocabulary available, the intended cart norm
("a cart one picks up must be returned") and a coordinate-anchored alternative ("every interaction must be
followed by an interaction at the return location") explain ordinary shopping demonstrations equally well. After
five demonstrations the posterior favoured the alternative 0.75 to 0.20. Each further ordinary demonstration
added only 0.13 nats of evidence through the size principle, so the intended norm overtook the alternative only
after about 15 demonstrations. An intuitively diagnostic demonstration (return, reacquire and return again)
added the same 0.13 nats. Under the maximum-entropy model, a demonstration that complies with both hypotheses is
informative only through its task context. Choosing the context that maximised the exact expected information
between the two leading hypotheses (an item next to an already-parked cart) produced a single demonstration that
moved the posterior of the intended norm from 0.20 to 1.00 and held-out accuracy from 0.80 to 1.00. Fifty
randomly chosen contexts achieved this only 7 times. Norm discovery can therefore be limited by which contexts
are observed rather than by how many demonstrations are seen, and posterior uncertainty can be used to request
the demonstrations that resolve it.

---

## Files

Raw and summary results (CSV / JSON / JSONL):
* `results/deterministic_proposals/reconstruction/`: `report.json`, `RECONSTRUCTION_REPORT.md`, `sanity_checks.json`,
  `original_results.csv`, `original_runs.jsonl` (with traces), `randomized_original_runs.jsonl`,
  `randomized_original_summary.csv`, `mlci_epsilon_sensitivity.csv`, `metadata.json`
* `results/deterministic_proposals/threshold_sensitivity/`: `runs.jsonl` (per-run, with traces), `runs.csv`, `traces.csv`
  (p_t, b_t, p_t/b_t), `summary.csv`, `running_min_diagnostics.json`, `metadata.json`
* `results/deterministic_proposals/noise_robustness/`: `runs.jsonl` (per seed, including full training datasets), `runs.csv`,
  `summary.csv`, `metadata.json`
* `results/deterministic_proposals/identifiability/`: `prefix_posteriors.csv`, `sequences.jsonl`, `context_scores_round1.csv`,
  `mi_landscape.csv`, `extra_demo_comparison.csv`, `extra_demo_summary.csv`, `long_control.csv`, `summary.json`,
  `metadata.json`
* `results/deterministic_proposals/supplement_dl_coding/`: `runs.jsonl`, `summary.csv`, `metadata.json`
* `results/deterministic_proposals/tables/`: `ALL_TABLES.md` plus one `.md` per experiment

Figures (`figures/deterministic_proposals/`, PNG + PDF):
* `gridworld_cart`, `gridworld_aisle`
* `threshold_accuracy`, `threshold_f1`, `threshold_refinements`, `threshold_p_intended`,
  `threshold_unseen_accuracy`, `threshold_traces`, `threshold_logp_vs_length`
* `noise_behavioural`, `noise_violation`. In the unseen-accuracy panel, fixed and MLCI coincide at 0.40.
* `identifiability_posterior`, `identifiability_random_vs_diagnostic`
