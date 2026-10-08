# Description-length prior ablation for Experiment 3 (layout-independent norm proposal)

**Question.** Does a syntactically consistent description-length prior, in which `at(x,y)` costs 3 units (predicate
+ two integer constants) like every other integer-bearing expression, let Bayesian inference select the
layout-independent norm GPT-5.5 already proposes? This uses no new LLM calls and no new demonstrations.

**Short answer.** It helps only marginally, and not in the way hoped.
* Token-consistent coding raises posterior support for intended-equivalent norms in every condition with a correct
  proposal (C2: +0.12 and +0.10).
* It removes the spurious coordinate MAP in most single-layout runs.
* But it makes the intended norm the MAP in **only 1 of 20** single-layout (C2) runs. Most freed runs instead select
  *other, non-equivalent* relational rules, some of which over-generalise (false positives on L3).

Code: `experiments/llm_layout_generalization_dl_ablation.py` (one stage, no API).
Outputs: `results/llm_layout_generalization_dl_ablation/`.

## Design
* **The only varied factor is the description-length coding** in the prior
  P(h | H) ∝ exp(−λ L(α) − γ L(N)), with λ = γ = 0.2 and joint normalisation over the candidate set:
  * **nodes** (original): atoms 1 (including `at(x,y)`), operators 1, numeric comparisons 3;
  * **tokens**: the existing `dl_coding="tokens"` implementation (`abstractions/base.py::formula_size`), identical
    except that `at(x,y)` / `cartAt(x,y)` cost 3.
* **Fix to the original pipeline's call path.** `multi_posterior()` called `L_alpha()` / `L_norm()` without a coding
  argument, so it always used node coding. The ablation script therefore computes the prior explicitly with
  `L_alpha(coding)` / `L_norm(coding)`. The original module is not modified.
* **Candidate sets** are rebuilt exactly as in `llm_layout_generalization.py::analyze`: the initial GPT-5.5 α₀ set,
  then the valid refined GPT-5.5 proposals (same validation), then the two spurious coordinate competitors, in that
  order (ties break identically). Source: the saved proposals of the original experiment
  (`results/llm_layout_generalization/`) and the counterfactual-guidance experiment
  (`results/llm_layout_generalization_counterfactual/`).
  * C1 is identical in both experiments, so it is evaluated once and listed once.
* **Unchanged:** the training demonstrations (L1 D1–D3; plus L2 B1–B3 in C3), the max-ent likelihood with the
  per-layout partition functions, the W grid with a uniform w prior, the probe sets, the L1 held-out set and the 10 L3
  diagnostics.
* **Odds decomposition** (C2, every run with an intended-equivalent proposal):
  log P(h_c|D)/P(h_x|D) = [log P(D|h_c) − log P(D|h_x)] + [log P(h_c) − log P(h_x)],
  with P(D|h) = Σ_w P(D|h,w)/|W|.
  * h_x is the run's original (node-coding) MAP: always `SPUR_memorised_cells` in C2.
  * h_c is the **highest-posterior intended-equivalent hypothesis under the coding being analysed**. It is chosen per
    coding; the same hypothesis was chosen under both codings in every C2 run.
  * Likelihoods include each hypothesis's own partition function, so they are *not* assumed equal just because both
    hypotheses have zero training violations.

## Validation (`validation.json`)
* **Node coding reproduces the saved Experiment 3 results exactly** in all 50 run-conditions (MAP id, selection,
  P(intended), L1 and L3 accuracy).
* **The likelihood is identical under both codings.** The full log P(D|h,w) matrix was computed independently for
  each coding and compared, with exact equality. Only the prior and posterior differ.
* **The saved GPT-5.5 proposal transcripts are unchanged** (SHA-256 before and after).
* **No API calls:** the API key was removed from the process, and the response cache had 471 files before and
  after.
* **Both spurious competitors are retained** in every candidate set (four variants listed).
* **The L3 diagnostic trajectories match `setup.json`, and the probe-set files are unchanged.**
* **Description lengths match hand calculation:**

  | Hypothesis | Nodes L(α) / L(N) | Tokens L(α) / L(N) |
  |---|---|---|
  | `SPUR_memorised_cells`, 4 cells | 0 / 8 | 0 / 16 |
  | `SPUR_aisle_region_coords`, 9 cells | 0 / 13 | 0 / 31 |
  | intended `S_inAisle` | 7 / 4 | 7 / 4 |

## Results: node vs token coding (10 runs per row)

| Guidance | Condition | Prior | Correct proposals | Correct MAP | Spurious MAP | Mean P(intended) | L1 acc. (unseen) | L3 acc. | L3 F1 | All L3 correct | L3 FPs |
|---|---|---|---|---|---|---|---|---|---|---|---|
| — | C1 | nodes | 0/10 | 0/10 | 9/10 | 0.000 | 0.73 (0.46) | 0.52 | 0.06 | 0/10 | 0 |
| — | C1 | tokens | 0/10 | 0/10 | 0/10 | 0.000 | 1.00 (1.00) | 0.66 | 0.55 | 0/10 | 4 |
| original | C2 | nodes | 6/10 | 0/10 | 10/10 | 0.077 | 0.70 (0.40) | 0.50 | 0.00 | 0/10 | 0 |
| original | C2 | tokens | 6/10 | 1/10 | 4/10 | 0.197 | 0.88 (0.76) | 0.65 | 0.43 | 1/10 | 6 |
| original | C3 | nodes | 6/10 | 5/10 | 0/10 | 0.245 | 1.00 (1.00) | 0.90 | 0.92 | 5/10 | 10 |
| original | C3 | tokens | 6/10 | 5/10 | 0/10 | 0.315 | 1.00 (1.00) | 0.90 | 0.92 | 5/10 | 10 |
| counterfactual | C2 | nodes | 6/10 | 0/10 | 10/10 | 0.045 | 0.70 (0.40) | 0.50 | 0.00 | 0/10 | 0 |
| counterfactual | C2 | tokens | 6/10 | 0/10 | 7/10 | 0.148 | 0.79 (0.58) | 0.54 | 0.16 | 0/10 | 2 |
| counterfactual | C3 | nodes | 7/10 | 5/10 | 0/10 | 0.236 | 1.00 (1.00) | 0.90 | 0.92 | 5/10 | 10 |
| counterfactual | C3 | tokens | 7/10 | 5/10 | 0/10 | 0.290 | 1.00 (1.00) | 0.90 | 0.92 | 5/10 | 10 |

Paired differences (token − node; same candidate sets and data; 95% paired bootstrap CI over runs):

| Guidance | Condition | Δ P(intended) tokens − nodes | Δ L3 acc. | Correct MAP gained / lost |
|---|---|---|---|---|
| — | C1 | +0.00 [+0.00, +0.00] | +0.14 [+0.08, +0.20] | 0 / 0 |
| original | C2 | +0.12 [+0.05, +0.20] | +0.15 [+0.05, +0.26] | 1 / 0 |
| original | C3 | +0.07 [+0.02, +0.12] | +0.00 [+0.00, +0.00] | 0 / 0 |
| counterfactual | C2 | +0.10 [+0.05, +0.16] | +0.04 [+0.00, +0.10] | 0 / 0 |
| counterfactual | C3 | +0.05 [+0.02, +0.10] | +0.00 [+0.00, +0.00] | 0 / 0 |

## C2 (single layout) per run: why the correct norm loses or wins

| Guidance | Run | Correct proposed | Node-prior MAP | Token-prior MAP | P(int.) nodes → tokens | Δ log-lik. | Δ log-prior nodes → tokens | Log odds nodes → tokens | L3 acc. nodes → tokens | L3 FPs (tokens) |
|---|---|---|---|---|---|---|---|---|---|---|
| original | 0 | yes | `SPUR_memorised_cells` | `SPUR_memorised_cells` (coordinate) | 0.025 → 0.091 | 0.005 | -3.2 → -1.6 | -3.195 → -1.595 | 0.5 → 0.5 | 0 |
| original | 1 | no | `SPUR_memorised_cells` | `g0_no_cart_in_single_file_passage` (relational) | 0.000 → 0.000 | — | — → — | — → — | 0.5 → 0.8 | 2 |
| original | 2 | yes | `SPUR_memorised_cells` | `g0_no_cart_in_single_file_aisle` (relational) | 0.212 → 0.389 | 0.005 | -0.6 → 1.0 | -0.595 → 1.005 | 0.5 → 0.8 | 2 |
| original | 3 | no | `SPUR_memorised_cells` | `g0_push_carts_only_where_aisle_is_wide_enough` (relational) | 0.000 → 0.000 | — | — → — | — → — | 0.5 → 0.5 | 2 |
| original | 4 | no | `SPUR_memorised_cells` | `g0_cart_free_single_file_shelf_aisles` (relational) | 0.000 → 0.000 | — | — → — | — → — | 0.5 → 0.7 | 0 |
| original | 5 | no | `SPUR_memorised_cells` | `g0_no_cart_in_single_file_shelf_aisle` (relational) | 0.000 → 0.000 | — | — → — | — → — | 0.5 → 0.7 | 0 |
| original | 6 | yes | `SPUR_memorised_cells` | `SPUR_memorised_cells` (coordinate) | 0.094 → 0.277 | 0.005 | -1.8 → -0.2 | -1.795 → -0.195 | 0.5 → 0.5 | 0 |
| original | 7 | yes | `SPUR_memorised_cells` | `SPUR_memorised_cells` (coordinate) | 0.069 → 0.250 | 0.005 | -2.2 → -0.6 | -2.195 → -0.595 | 0.5 → 0.5 | 0 |
| original | 8 | yes | `SPUR_memorised_cells` | `g0_no_carts_between_facing_shelves` (relational) | 0.273 → 0.658 | 0.005 | -0.6 → 1.0 | -0.595 → 1.005 | 0.5 → 1.0 | 0 |
| original | 9 | yes | `SPUR_memorised_cells` | `SPUR_memorised_cells` (coordinate) | 0.095 → 0.303 | 0.005 | -1.8 → -0.2 | -1.795 → -0.195 | 0.5 → 0.5 | 0 |
| counterfactual | 0 | yes | `SPUR_memorised_cells` | `SPUR_memorised_cells` (coordinate) | 0.100 → 0.350 | 0.005 | -1.8 → -0.2 | -1.795 → -0.195 | 0.5 → 0.5 | 0 |
| counterfactual | 1 | yes | `SPUR_memorised_cells` | `SPUR_memorised_cells` (coordinate) | 0.096 → 0.296 | 0.005 | -1.8 → -0.2 | -1.795 → -0.195 | 0.5 → 0.5 | 0 |
| counterfactual | 2 | no | `SPUR_memorised_cells` | `g0_no_cart_motion_in_single_file_clearance` (relational) | 0.000 → 0.000 | — | — → — | — → — | 0.5 → 0.5 | 2 |
| counterfactual | 3 | yes | `SPUR_memorised_cells` | `SPUR_memorised_cells` (coordinate) | 0.066 → 0.219 | 0.005 | -2.2 → -0.6 | -2.195 → -0.595 | 0.5 → 0.5 | 0 |
| counterfactual | 4 | yes | `SPUR_memorised_cells` | `SPUR_memorised_cells` (coordinate) | 0.048 → 0.189 | 0.005 | -2.6 → -1.0 | -2.595 → -0.995 | 0.5 → 0.5 | 0 |
| counterfactual | 5 | yes | `SPUR_memorised_cells` | `SPUR_memorised_cells` (coordinate) | 0.046 → 0.171 | 0.005 | -2.6 → -1.0 | -2.595 → -0.995 | 0.5 → 0.5 | 0 |
| counterfactual | 6 | no | `SPUR_memorised_cells` | `g0_no_cart_in_shelf_canyon` (relational) | 0.000 → 0.000 | — | — → — | — → — | 0.5 → 0.7 | 0 |
| counterfactual | 7 | yes | `SPUR_memorised_cells` | `SPUR_memorised_cells` (coordinate) | 0.091 → 0.254 | 0.005 | -1.8 → -0.2 | -1.795 → -0.195 | 0.5 → 0.5 | 0 |
| counterfactual | 8 | no | `SPUR_memorised_cells` | `g0_no_cart_in_single_file_shelf_aisle` (relational) | 0.000 → 0.000 | — | — → — | — → — | 0.5 → 0.7 | 0 |
| counterfactual | 9 | no | `SPUR_memorised_cells` | `SPUR_memorised_cells` (coordinate) | 0.000 → 0.000 | — | — → — | — → — | 0.5 → 0.5 | 0 |

Δ values compare the best intended-equivalent hypothesis with the coordinate rule `SPUR_memorised_cells`. They are
blank when no intended-equivalent norm was proposed in that run.

## Diagnosis
* **The coordinate rule wins only because of its prior.** In every C2 run with a correct proposal, the likelihood
  slightly *favours* the correct norm, by +0.005 nats (it also forbids the unvisited aisle A3, a tiny size-principle
  gain). So the likelihood is essentially uninformative between them.
  * The log-prior difference is −0.2 × (L_correct − L_coordinate). Under node coding L_coordinate = 8, while the
    correct GPT-5.5 norms have total length 11, 17, 19, 21 or 24, giving −0.6 to −3.2 nats.
* **Token coding shifts every comparison by exactly +1.6 nats** (L_coordinate 8 → 16). It **reverses the pairwise
  ranking only when the correct norm has length ≤ 15.** In C2 that happens in 2/12 comparisons: original runs 2 and
  8, both with L = 11, log odds −0.60 → +1.00.
  * For the other 10 comparisons the correct norm is still too complex (lengths 17–24). Its support rises (e.g. run 9:
    0.10 → 0.30), but it does not become the MAP.
* **A reversed pairwise ranking does not guarantee a correct MAP.**
  * Run 8 (original): the intended norm becomes the MAP; L3 accuracy 0.50 → 1.00.
  * Run 2 (original): another relational but **non-equivalent** rule ("single-file aisle", width-based) outranks
    it. It produces 2 false positives on L3.
* **Coordinate-free MAPs that are not the intended norm.** Token coding removes the spurious coordinate MAP in 6/10
  (original) and 3/10 (counterfactual) C2 runs, and in all 9 C1 runs.
  * Most replacements are relational but non-equivalent. Width-based definitions over-flag the compliant L3 trap
    paths: 6 and 2 false positives in C2, 4 in C1.
  * In C1, the vertical-only "shelves east and west" definitions get L1 right (1.00) but miss L3's horizontal aisle
    (L3 0.66).
* **C3 (two layouts) is unaffected at the MAP level** (5/10 in both experiments, identical L3 metrics). Support for
  intended-equivalent norms rises modestly (+0.07 and +0.05).
* **Counterfactual guidance + token prior is no better than original guidance + token prior.** Its correct
  proposals are longer (lengths 17–21), so none crosses the length-15 threshold.

## What this establishes, and what it does not
* **Establishes:**
  * With only one layout, the likelihood cannot separate the coordinate rule from the intended norm (Δ ≈ 0.005 nats).
    Selection is decided entirely by the description-length prior.
  * The original node coding makes coordinate memorisation cheap. Consistent token coding removes most spurious
    coordinate MAPs and raises support for the intended norm.
  * But because GPT-5.5's correct definitions are usually verbose, consistent coding rarely makes them the MAP
    (1/20 C2 runs). Its main effect is to move the MAP to *other* relational candidates, which are not always correct
    and sometimes over-generalise.
* **Does not establish:**
  * that token coding is "the right" prior (it was chosen for consistency, not tuned);
  * effects for other λ/γ;
  * anything beyond these 10 runs and this domain.
  * Rates of 1/10 vs 0/10 are not statistically distinguishable.
  * The C3 confound remains: layout diversity is confounded with twice as many demonstrations.

## Files
* `runs.jsonl` / `runs.csv`: per guidance × condition × run × coding. Includes the MAP's description, L(α), L(N),
  weighted penalty, posterior, P(intended), L1/L3 metrics and false positives.
* `odds_decomposition.csv`: likelihood and prior components, both codings.
* `c2_per_run.csv`, `summary.csv`, `paired_differences.csv`, `validation.json`.
* `table.tex`, `paragraph.tex`.
