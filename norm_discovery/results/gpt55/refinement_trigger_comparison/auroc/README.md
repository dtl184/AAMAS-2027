# Refinement-trigger AUROC (actual GPT-5.5 counterfactual refinement labels)

## Completing the labels
* The original labelling stage attempted 380 opportunities (domain × protocol × run × t): 308 succeeded and 72
  failed. All 72 failures were aisle 20-demonstration opportunities, failing with HTTP 429 "no credits remaining".
  These counts were verified against `labels.jsonl` before resuming.
* `python -m experiments.gpt55_trigger_comparison labels-resume` reran **only** the 72 failed keys.
  * A cache-only dry run first confirmed 72 needed new calls and 308 were preserved.
  * All 72 succeeded: 72 fresh GPT-5.5 calls, all `gpt-5.5-2026-04-23`, 676k input and 435k output tokens.
* `labels.jsonl` was **not modified** (SHA-256 verified).
* `labels_complete.jsonl` contains the 308 original successful lines copied byte-for-byte (verified), followed by
  the 72 new records. It has 380 unique keys and 0 errors. The new records alone are in `labels_resumed.jsonl`.
* The labels are counterfactual: "useful at t" means that adding GPT-5.5's refinement of D₁…D_t raises
  P(intended-equivalent) by ≥ 0.05, or raises held-out or unseen-aisle accuracy, compared with not refining at t.
  This definition was fixed in the original stage and is used only for evaluation.

## AUROC protocol (`experiments/trigger_auroc.py`)
* The statistics are computed on the *unrefined* learner path (`paths.jsonl`), at every opportunity t ≥ 3:
  * raw: log p_t
  * length-normalised: log p_t / |τ_t|
  * task-cost-adjusted: log p_t + β·C(τ_t)
  * norm-relative: log p_t − log P_task(τ_t)
* A trigger fires on low values, so AUROC = P(score of a useful opportunity < score of a not-useful one), with ties
  counted as ½.
* Each statistic is scored in two forms:
  * **value**: s_t itself;
  * **margin**: s_t − min_{i<t} s_i, the quantity the running-minimum trigger thresholds.
* 95% CIs use a run-clustered bootstrap (4,000 replicates). Opportunities within a run are dependent, so whole runs
  are resampled.
* **Reportability rule, fixed before computing any AUROC:** the cell's labels must be complete, with ≥ 10 useful and
  ≥ 10 not-useful opportunities.

| Cell | Opportunities | Useful / not useful | Reportable? |
|---|---|---|---|
| cart, 20-demo pools | 180 | 142 / 38 | **yes** |
| aisle, 20-demo pools | 180 | 177 / 3 | no: only 3 negatives |
| cart, original 3 demos | 10 | 6 / 4 | no |
| aisle, original 3 demos | 10 | 10 / 0 | no: AUROC undefined |
| both domains pooled, 20-demo | 360 | 319 / 41 | passes the count rule, but **confounded by domain** (base rate of usefulness 98% vs 79%); not interpretable as detection |

## Results: cart, 20-demonstration pools (the only reportable single-domain cell)

| Statistic | AUROC, value [95% CI] | AUROC, margin [95% CI] | Spearman with demo length (value) |
|---|---|---|---|
| A. raw predictive | 0.55 [0.42, 0.68] | 0.50 [0.37, 0.65] | −0.75 |
| B. length-normalised | 0.54 [0.40, 0.70] | 0.52 [0.41, 0.63] | −0.53 |
| C. task-cost-adjusted | 0.60 [0.48, 0.73] | 0.49 [0.40, 0.61] | +0.52 |
| D. norm-relative | 0.61 [0.49, 0.74] | 0.50 [0.39, 0.63] | −0.24 |

**Every CI includes 0.5.** In the cart domain, none of the four statistics separates useful from not-useful
refinement opportunities better than chance, in either form. The margin form, which is what the trigger actually
uses, is at 0.49–0.52 for all four. The point estimates for the value form are ordered norm-relative ≥
task-adjusted > raw ≈ length-normalised, but the differences are well within the uncertainty.

## Non-reportable cells (descriptive only)
* **Aisle, 20-demo:** refinement was useful at 177/180 opportunities. The α₀ representation is inadequate almost
  everywhere, so the detection question is close to degenerate: an always-on trigger would be "correct" 98% of the
  time. The AUROCs (e.g. task-adjusted margin 0.06, raw value 0.74) rest on 3 negative opportunities, and the
  run-clustered CIs understate that uncertainty. They should not be reported.
* **Original 3-demonstration protocol:** one opportunity per run (10 per domain). The aisle cell has no negatives.
* **Pooled across domains:** the norm-relative value reaches 0.78, but this mostly reflects the domains themselves.
  Aisle opportunities have low norm-relative scores (α₀ has almost no advantage over task cost there) and are nearly
  all useful. It is not evidence of detection within a domain.

## What can be reported in the paper
* **Can be reported:**
  * The length/task-cost confound of each statistic: the Spearman correlations above, and the full table in
    `../confound_table.csv` once `analyze` runs (the paths are complete).
  * Cart 20-demo AUROCs, with CIs: **no statistic detects useful refinement opportunities better than chance.**
  * The aisle base rate: refinement useful at 98% of opportunities.
* **Should not be reported as findings:**
  * aisle AUROCs (3 negatives);
  * 3-demonstration AUROCs (n = 10);
  * the pooled cross-domain AUROC as evidence of detection;
  * any claim that one statistic is a better *detector* than another.
* **Implication:** the case for task-adjusted or norm-relative over raw rests on **robustness to trajectory length
  and inefficiency** (Selected-Trigger Noise Robustness; confound tables), not on better identification of useful
  refinement moments.

## Files
* `auroc.csv` / `auroc.json`: all cells, both forms, with CIs and counts.
* `cell_status.csv`: completeness and reportability per cell.
* `opportunities.csv`: per-opportunity statistics and labels.
* Figure: `figures/gpt55/refinement_trigger_comparison/auroc_roc_curves.png`.
