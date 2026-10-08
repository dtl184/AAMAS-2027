# Selected-trigger noise robustness (frozen GPT-5.5 proposals, no API calls)

Run: `cd /home/train/norm_discovery && /home/train/anaconda3/bin/python -m experiments.noise_selected_trigger`

Files: runs.jsonl / runs.csv (per run, including per-t trigger events), trigger_events.csv, summary.csv, paired_differences.csv, trigger_response_by_category.csv, aggregate.json, mlci_reuse_check.json, metadata.json.
**What this evaluates.** Trigger robustness plus Bayesian-inference robustness under imperfect demonstrations,
using **frozen real GPT-5.5 proposal sets**.
* **No new OpenAI calls were made.** The process ran without the API key, and the response cache held 431 files
  before and after.
* Each replicate replays one of the 10 existing GPT-5.5 runs per domain from
  `results/gpt55/refinement_threshold_extended` (ρ = 20,000). Each contributes its initial set (8 hypotheses) and
  its refined set (8 hypotheses), both proposed from the *clean original* demonstrations.
* **This does not test whether GPT-5.5 would propose different hypotheses after seeing noisy demonstrations.**
* After the first trigger, further triggers add no new hypotheses. They are counted as refinement events but are
  no-ops.

**Trigger.** The trigger comparison never selected a final trigger: it is incomplete, since its labels and runs
stages stopped when credits ran out. Per the protocol:
* Primary: the task-cost-adjusted score log p_t + β·C(τ_t), with the quantile rule s_t < Q_0.10 (accepted scores).
  q = 0.10 was fixed a priori.
* Sensitivity: q = 0.05 and q = 0.20.
* Secondary: the norm-relative trigger at q = 0.10.
* Comparison: the old raw trigger, p_t < 0.1·b_t, run on the same frozen proposals.

**Protocol.**
* Both domains; the same environments, held-out sets, abstraction language, priors and inference.
* 20 seeds × 20 demonstrations. These are the same datasets as `results/deterministic_proposals/noise_robustness`:
  the same clean sequence per seed for every method and q, with corrupted demonstrations chosen per seed and nested
  across q.
* q ∈ {0, 0.05, 0.10, 0.20, 0.30}; behavioural (compliant) and norm-violation noise.
* 2,520 inference runs. MLCI's 360 runs are reused from the deterministic experiment (identical datasets and
  settings; recomputing 3 of them gave identical results).
* Values below are means [95% bootstrap CI over seeds]. Method differences are paired by seed
  (`paired_differences.csv`).

**Held-out accuracy at q = 0 → q = 0.30.** Adaptive and oracle are identical in every seed, so their curves overlap.

| Domain | Noise | Adaptive (task, q=0.10) | Old raw trigger | Fixed | Oracle | MLCI |
|---|---|---|---|---|---|---|
| cart | behavioural | 0.92 [0.88,0.96] → 0.92 | 0.87 → 0.92 | 0.80 → 0.80 | 0.92 → 0.92 | 0.79 → 0.82 |
| cart | violation | 0.92 → 0.92 | 0.87 → 0.87 | 0.80 → 0.80 | 0.92 → 0.92 | 0.79 → **0.42** [0.40,0.47] |
| aisle | behavioural | 0.99 [0.97,1.00] → 1.00 | 0.82 → 0.92 | 0.60 → 0.60 | 0.99 → 1.00 | 0.59 → 0.69 |
| aisle | violation | 0.99 → 1.00 | 0.82 → 0.78 | 0.60 → 0.59 | 0.99 → 1.00 | 0.59 → **0.40** |

* Aisle unseen accuracy: adaptive and oracle stay at 1.00 at every q, for both noise types. The old raw trigger gets
  0.73 → 0.67 under violation noise; fixed and MLCI stay at 0.40.
* Violation F1 follows accuracy. MLCI drops from 0.74 to 0.05 (cart) and from 0.50 to 0.00 (aisle).

**Refinement triggers per 20-demonstration run** (out of 18 opportunities). Behavioural noise is first.

| Trigger | Cart, behavioural, q=0 → 0.30 | Cart, violation, q=0 → 0.30 | Aisle, behavioural, q=0 → 0.30 | Aisle, violation, q=0 → 0.30 |
|---|---|---|---|---|
| Task-adjusted, q=0.10 (primary) | 5.70 → 5.70 (paired Δ 0.00 [0.00,0.00]) | 5.70 → 7.50 (Δ +1.80 [0.40,3.10]) | 9.00 → 9.00 (Δ 0.00) | 9.00 → 8.65 |
| Norm-relative, q=0.10 | 5.60 → 5.60 | 5.60 → 7.45 | 5.35 → 5.75 | 5.35 → 6.80 |
| Old raw, ρ=0.1 | 1.50 → **4.70** | 1.50 → 2.35 | 2.15 → **4.25** | 2.15 → 1.50 |

**Trigger response by demonstration type.** Scores come from the fixed-abstraction runs and are measured in
z-units relative to the same run's clean demonstrations. The firing rate is that of the adaptive runs at t ≥ 3.

| Domain | Statistic | Behavioural (compliant) | Norm-violating | Firing rate: clean / behavioural / violating |
|---|---|---|---|---|
| cart | raw | −6.55 [−7.39, −5.67] | +1.38 [1.02, 1.73] | 0.09 / **0.78** / 0.14 |
| cart | task-adjusted | −0.02 [−0.36, 0.29] | −7.19 [−8.35, −5.98] | 0.31 / 0.28 / **0.85** |
| cart | norm-relative | 0.00 [−0.26, 0.24] | −21.08 [−24.24, −17.84] | 0.30 / 0.30 / **0.88** |
| aisle | raw | −8.80 [−9.83, −7.74] | −0.27 [−0.37, −0.18] | 0.11 / **0.79** / 0.19 |
| aisle | task-adjusted | −0.11 [−0.37, 0.17] | +0.35 [0.20, 0.51] | 0.48 / 0.51 / 0.69 |
| aisle | norm-relative | −0.03 [−0.33, 0.33] | −1.63 [−1.84, −1.41] | 0.29 / 0.34 / **0.84** |

**Inferred violation cost w (MAP hypothesis, adaptive), q = 0 → 0.30 under violation noise:** cart 22.4 → 13.2;
aisle 15.7 → 1.0. Under behavioural noise w is unchanged (cart 22.4, aisle 15.7).

**Answers.**
1. **Behavioural noise does not reduce held-out accuracy** for adaptive or oracle (paired Δ = 0.00, or +0.01 in the
   aisle domain). It does not hurt fixed either.
2. **The selected trigger does not refine more under behavioural noise.** Its trigger count is exactly unchanged
   (paired Δ 0.00 [0.00, 0.00] in both domains), and compliant detours score the same as clean demonstrations
   (−0.02 and −0.11 z). This is exact rather than approximate: when a detour violates no candidate hypothesis,
   log p + β·C depends only on the task context, not on the path. However, the trigger refines often *regardless
   of noise*: 5.7 (cart) and 9.0 (aisle) triggers per run, and 31–48% of clean demonstrations. Most of these
   triggers are unnecessary.
3. **Yes, against the inefficiency confound.** The old raw trigger fires on 78–79% of behavioural demonstrations
   versus 9–11% of clean ones. Its refinement count rises from 1.5 to 4.7 (cart) and from 2.15 to 4.25 (aisle) with
   behavioural noise. It is also less accurate: aisle 0.82 vs 0.99 at q = 0; paired Δ +0.22 [0.14, 0.30] in favour
   of the selected trigger at 30% violation noise. Its gains under behavioural noise (aisle 0.82 → 0.92) come from
   the confound, which happens to make it refine.
4. **With these frozen GPT-5.5 hypothesis sets, genuine violation noise up to 30% does not reduce accuracy**
   (cart 0.92, aisle 1.00). The posterior responds by lowering the violation cost w (cart 22 → 13, aisle 16 → 1)
   rather than abandoning the norm. Because held-out classification uses V ≥ 1 regardless of w, accuracy is
   preserved even though the inferred norm becomes very "soft" in the aisle domain. Contrast the
   deterministic-proposal ablation, where the oracle dropped from 1.00 to 0.77 (cart) and from 0.92 to 0.55 (aisle).
   There, a near-equivalent coordinate competitor and a "near any shelf" competitor absorbed the posterior. The
   robustness seen here therefore depends on the candidate set and is not a general property of the inference.
5. **Oracle and adaptive are identical.** Paired difference: 0.00 in every condition. The adaptive learner refines
   early (first trigger at t ≈ 4–5) and receives the same frozen refined set the oracle has from the start.
6. **MLCI degrades steeply under genuine violations:** cart 0.79 → 0.42 (F1 0.74 → 0.05), aisle 0.59 → 0.40
   (F1 → 0.00). Paired advantage of adaptive over MLCI at 30%: +0.49 [0.44, 0.54] (cart), +0.60 [0.60, 0.60] (aisle).
7. **Aisle unseen-structure generalisation survives both noise types** for adaptive and oracle: 1.00 at every q.
8. **Main robustness limitation: the trigger fires too often and cannot tell a violating demonstrator from an
   inadequate representation.**
   * It fires on 85–88% of norm-violating cart demonstrations (task-adjusted and norm-relative), and on about a
     third to a half of clean ones.
   * With live GPT-5.5 proposals each firing would be a paid refinement call: about 6–9 per 20-demonstration run.
     Every firing after the first is unnecessary here.
   * In the aisle domain the task-adjusted score barely reacts to violations (+0.35 z), because the α₀ coordinate
     hypotheses do not cover the violated region. The norm-relative score does react (−1.63 z).
   * Secondary limitation: violation robustness rests on (a) the particular GPT-5.5 candidate set and (b) a
     classification rule that ignores the shrinking w.
   * The frozen protocol also cannot show how GPT-5.5 itself responds to noisy demonstrations.

**Status of results.**
* Real frozen GPT-5.5 proposal results: everything in this section except MLCI. All conditions are complete
  (20/20 seeds).
* MLCI: no proposals involved; reused from the deterministic experiment.
* Deterministic-proposal ablation: Part II, Section 2 (`results/deterministic_proposals/noise_robustness/`).
* Incomplete / not run: the live GPT-5.5 noise experiment (Part I, I.3); the trigger-comparison labels and runs
  stages, so no trigger was formally selected.
