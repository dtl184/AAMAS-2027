# Experiment 1 on new demonstration sequences (seeds 100–119): three adaptive refinement policies

**Purpose.** Compare the three refinement policies on training sequences that Experiment 2 never used.

Code: `experiments/exp1_policy_heldout.py`. It reuses `experiments/noise_selected_trigger.py::_job` unchanged:
`FrozenProposalProvider` replays the cached GPT-5.5 proposal sets from
`results/gpt55/refinement_threshold_extended`, followed by `NormLearner` and the same evaluation. **No OpenAI calls**:
the key is removed from the process and the response cache is unchanged (`validation.json`).

* **What it evaluates:** trigger robustness plus Bayesian inference given *frozen* real GPT-5.5 proposal sets. It
  does not evaluate how GPT-5.5 would propose from these new sequences.
* **Policies:**
  * raw predictive trigger, ρ = 0.1;
  * task-cost-adjusted trigger, quantile q = 0.10;
  * norm-relative trigger, q = 0.10.
  * Fixed and oracle abstraction are included as references only.
* **Unchanged settings:** β = 2; λ = γ = 0.2; W = {0, 0.5, …, 30}; 20 demonstrations per run; both domains; frozen
  replicate = seed mod 10.
* **Seeds:** 20 per domain, indices 100–119 (Experiment 2 used 0–19).
* **Conditions:** clean (q = 0, primary), plus behavioural and violation noise at q ∈ {0.05, 0.10, 0.20, 0.30}. The
  original request was cut off after "Keep the"; I assumed the Experiment 2 conditions were meant to be kept.
* **Data independence (`data_independence_check.json`):**
  * 0 of 360 training sequences are identical to an Experiment 2 sequence, and no dataset seeds are shared.
  * Individual demonstrations can recur, because the task pool is small (14 cart / 10 aisle tasks) and optimal paths
    have few tie variants. In the clean condition, 17/400 cart and 124/400 aisle demonstrations also occur somewhere
    in Experiment 2.
* **Paired design:** verified identical demonstrations (including corrupted positions) across all methods for each
  (domain, seed, condition).

## Primary results: clean sequences (q = 0), mean [95% bootstrap CI over 20 seeds]

| Domain | Policy | Held-out acc. | Violation F1 | Unseen-aisle acc. | Runs refining | Refinements per run | First refinement t |
|---|---|---|---|---|---|---|---|
| cart | Raw (ρ=0.1) | 0.88 [0.84, 0.92] | 0.88 | – | 0.55 | 2.30 [0.55, 4.40] | 3.7 |
| cart | Task-adjusted (q=0.10) | 0.91 [0.87, 0.95] | 0.91 | – | 0.95 | 4.75 [3.05, 6.45] | 4.7 |
| cart | Norm-relative (q=0.10) | 0.91 [0.87, 0.95] | 0.91 | – | 0.95 | 4.85 [2.85, 6.95] | 5.0 |
| cart | *Fixed (ref.)* | 0.80 | 0.80 | – | 0 | 0 | – |
| cart | *Oracle (ref.)* | 0.92 [0.88, 0.96] | 0.92 | – | – | – | – |
| aisle | Raw (ρ=0.1) | 0.82 [0.74, 0.90] | 0.70 | 0.73 [0.61, 0.85] | 0.55 | 5.35 [2.65, 8.25] | 3.9 |
| aisle | Task-adjusted (q=0.10) | **1.00** | 1.00 | **1.00** | 1.00 | 7.15 [5.35, 9.20] | 4.2 |
| aisle | Norm-relative (q=0.10) | **1.00** | 1.00 | **1.00** | 1.00 | 5.55 [4.15, 7.00] | 4.6 |
| aisle | *Fixed (ref.)* | 0.60 | 0.33 | 0.40 | 0 | 0 | – |
| aisle | *Oracle (ref.)* | 1.00 | 1.00 | 1.00 | – | – | – |

**Paired differences (clean; `paired_differences.csv`):**
* Aisle: task-adjusted − raw = +0.18 [0.10, 0.26] accuracy and +0.27 [0.15, 0.39] unseen-aisle accuracy. Norm-relative
  − raw is identical.
* Cart: either new trigger − raw = +0.03 [−0.01, 0.07] accuracy, with the CI including 0.
* Norm-relative − task-adjusted: accuracy 0.00 in both domains; refinements −1.60 [−3.25, −0.30] in the aisle domain
  (fewer refinement calls) and +0.10 [−0.55, 0.75] in the cart domain.

**Under noise** (full table in `summary.csv`; figures `figures/exp1_policy_heldout/policies_*.png`):
* The two new triggers stay at oracle-level accuracy under both noise types. Cart: 0.91 at every q. Aisle: 0.98–1.00.
* The raw trigger stays below them in the aisle domain under violation noise: 0.84 at q = 0.3 vs 1.00 for
  task-adjusted, paired +0.16 [0.08, 0.24].
* Under behavioural noise the raw trigger catches up (aisle 0.97 at q = 0.3; cart 0.90). Long, inefficient
  demonstrations make it fire, and it refines in 95% and 80% of runs, versus 55% when clean. This reproduces the
  length confound seen in Experiment 2.

## Conclusions this replication supports
* **On new sequences, both proposed triggers beat the raw trigger in the aisle domain** (+0.18 accuracy, +0.27
  unseen-aisle accuracy, paired CIs excluding 0). They reach oracle-level performance because they refine in every
  run, whereas the raw trigger refines in only 55%.
* **In the cart domain** the gain over raw is small and not significant (+0.03, CI includes 0).
* **Task-adjusted and norm-relative are indistinguishable in accuracy.** Norm-relative uses significantly fewer
  refinements in the aisle domain.
* **All adaptive policies refine repeatedly (about 5–7 triggers per 20-demo run).** With frozen proposals, repeats
  add no hypotheses. With live GPT-5.5 each would be a paid call, which is the over-triggering limitation already
  noted for Experiment 2.
* These results agree qualitatively with the Experiment 2 runs at q = 0, so the policy ranking is not an artefact of
  reusing Experiment 2's sequences.

## Files
`runs.jsonl` / `runs.csv`, `trigger_events.csv`, `summary.csv`, `paired_differences.csv`,
`data_independence_check.json`, `validation.json`, `metadata.json`, `table.tex`.
