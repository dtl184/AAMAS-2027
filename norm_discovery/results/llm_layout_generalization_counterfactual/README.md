# Experiment 3 with counterfactual-environment guidance (GPT-5.5, aisle domain)

**Question.** Does explicitly asking GPT-5.5 to reason about hypothetical changes to the store help it discover
layout-independent norms, including when it has seen demonstrations from only one store (c2)?

**Answer from 10 runs per condition: no measurable improvement on any of the three outcomes** (proposal, selection,
generalisation), in either the single-layout or the two-layout condition.

Code: `experiments/llm_layout_generalization_counterfactual.py` (stages `prepare` → `run` → `analyze`). The original
experiment is `experiments/llm_layout_generalization.py`, with results in `results/llm_layout_generalization/`.
Those results were **not modified**: SHA-256 checksums of all 31 files were checked before and after.

## What changed, and what did not
* **The only change** is the guidance paragraph inserted into the c2 and c3 prompts. It was replaced by the
  counterfactual guidance, verbatim as specified (`setup.json` → `guidance_paragraph`; the previous text is in
  `guidance_paragraph_previous_experiment`). The new script imports the original module and calls its own
  `prepare` / `analyze`, overriding only the output directory and the guidance string.
* **Verified automatically** (`prompt_verification.json`) for all 10 runs:
  * c1 prompts are byte-identical to the original experiment;
  * c2/c3 prompts equal the original c2/c3 prompts after substituting the guidance paragraph;
  * c2 contains exactly the 3 L1 demonstrations and nothing from L2 or L3;
  * c3 contains 3 × L1 and 3 × L2 demonstrations;
  * the guidance contains no "aisle" / "inAisle" and no description of L3.
* **Unchanged:**
  * the paired initial (α₀) hypothesis sets per run;
  * model `gpt-5.5` with API-default generation settings;
  * the vocabulary and schema, 8 hypotheses per call;
  * layouts L1/L2/L3, the training demonstrations, the 10 L3 diagnostics, and the probe sets;
  * the two spurious coordinate competitors;
  * priors, likelihood and W grid;
  * all evaluation code.
  * c1 reuses the cached original responses; its numbers reproduce the original experiment exactly.
* **Fresh GPT-5.5 calls:** 20 (replicate keys `genlayout_cf/<c2|c3>/<r>`). All returned `gpt-5.5-2026-04-23`, with
  no API errors and no parse retries. Usage: 92.8k input and 127.2k output tokens.

## Results (10 runs per condition)

| | c1 single layout (unchanged) | c2 previous guidance | **c2 counterfactual** | c3 previous guidance | **c3 counterfactual** |
|---|---|---|---|---|---|
| Valid proposals using coordinates | 0.17 | 0.00 | 0.00 | 0.00 | 0.00 |
| Relational proposals | 0.75 | 0.67 | 0.69 | 0.64 | 0.66 |
| Intended-equivalent proposal (all layouts) | 0/10 | 6/10 | **6/10** | 6/10 | **7/10** |
| MAP selection of the intended norm | 0/10 | 0/10 | **0/10** | 5/10 | **5/10** |
| MAP is a spurious coordinate rule | 9/10 | 10/10 | **10/10** | 0/10 | **0/10** |
| Mean P(intended-equivalent \| D) | 0.00 | 0.08 | 0.05 | 0.25 | 0.24 |
| L1 held-out acc. / unseen-aisle acc. | 0.73 / 0.46 | 0.70 / 0.40 | **0.70 / 0.40** | 1.00 / 1.00 | **1.00 / 1.00** |
| L3 acc. / violation F1 | 0.52 / 0.06 | 0.50 / 0.00 | **0.50 / 0.00** | 0.90 / 0.92 | **0.90 / 0.92** |
| Runs classifying all L3 trajectories correctly | 0/10 | 0/10 | **0/10** | 5/10 | **5/10** |
| Runs whose MAP is an over-broad relational rule with false positives | 0/10 | 0/10 | 0/10 | 5/10 | **5/10** |
| L3 false positives (total), all from relational MAPs | 0 | 0 | 0 | 10 | **10** |

* Single-layout MAPs make only **false negatives** on L3. The spurious coordinate rule never fires there (48–50
  false negatives in total over 10 runs).
* Every false positive in c3 comes from the two compliant "trap" trajectories in L3. Both conditions behave the same
  way: 2 per affected run, none on L1.
* Per-run files: `runs.jsonl`, `proposals.csv`, `false_positives.csv`
  (`false_positives_previous_experiment.csv` applies the same analysis to the earlier runs). The side-by-side
  numbers are in `comparison_with_previous.csv`.

## Analysis (primary comparison: c2 previous vs c2 counterfactual, same L1 demonstrations)

**(a) Proposal.** The intended-equivalent proposal rate is 6/10 under both guidance versions.
* The successes fall on different runs: 2 runs succeed under both, 4 only under the old guidance, and 4 only under
  the new.
* Run-to-run sampling variability therefore dominates any guidance effect. With n = 10, the experiment cannot detect
  a difference of this size.
* Both guidance versions remove coordinate-based proposals entirely; the counterfactual text adds nothing beyond
  that.
* In c3 the rate is 7/10 vs 6/10, a one-run difference.

**(b) Selection.** Counterfactual guidance does **not** help selection: 0/10 in c2, with the spurious memorised-cells
rule selected in 10/10 runs, as before.
* The reason is structural. With demonstrations from one layout, the coordinate rule ("no cart in the 4 cells the
  demonstrators avoided", description length 8) and the intended norm explain the training data essentially equally
  well. The description-length prior then favours the shorter rule.
* The intended-equivalent proposals produced under counterfactual guidance are, if anything, *longer*: minimum
  length 17 and median 18, against 11 and 17 previously. That makes them even less competitive. Mean P(intended)
  fell from 0.08 to 0.05.
* Better proposals cannot overcome this without either evidence that separates the hypotheses (a second layout) or a
  different prior. Neither prior nor competitor set was changed here.

**(c) Generalisation.** The selected MAP in c2 is the coordinate rule in every run, so L3 performance is unchanged at
chance (0.50, F1 0.00, 0/10 runs fully correct).
* In c3 generalisation is identical to the previous guidance (L3 0.90 / F1 0.92, 5/10 runs fully correct).
* The prompt now explicitly warns against "overly broad definitions". Even so, the remaining c3 failures are again
  over-broad relational MAPs: "single-file next to a shelf", defined with free width ≤ 1. These flag the two
  compliant trap paths in L3 as violations.

## What this experiment establishes, and what it does not
* **Establishes:** with this prompt pipeline, candidate set, prior and data, replacing the generalisation guidance by
  counterfactual-environment guidance did not improve proposal, selection or generalisation with GPT-5.5. A single
  layout's demonstrations still lead the Bayesian learner to a cheaper coordinate rule, even when the intended norm
  is among the proposals.
* **Does not establish:**
  * that counterfactual prompting is ineffective in general (10 runs per condition, one domain, one phrasing);
  * that the difference between the two guidance texts is exactly zero (proposal rates of 6/10 vs 6/10 and 7/10 vs
    6/10 are compatible with modest effects either way);
  * anything about other models or temperature settings. The API rejects `temperature` for `gpt-5.5`, so the API
    default was used.
* **Confound:** c3 still adds both a second layout *and* three demonstrations (6 vs 3), so environmental diversity
  is not separated from demonstration quantity. L2 also introduces horizontal aisles, which no single-layout
  condition sees.
* **Caveat:** every prompt lists GPT-5.5's own earlier α₀ proposals as the current hypotheses; some are named e.g.
  `no_cart_in_item_aisle`. This text is identical across conditions and experiments.

## Files
* `setup.json`: layouts, demonstrations, diagnostics, competitors, both guidance texts, and what varies.
* `prompts.jsonl`: c1/c2/c3 prompts. `prompt_verification.json`: the prompt-equality checks.
* `llm/<c>/run_XX/transcript.json`: prompt, raw response, parsed hypotheses, model, request id, usage.
* `runs.jsonl` / `runs.csv`, `proposals.csv`, `summary.csv`, `false_positives*.csv`, `comparison_with_previous.csv`.
* `table.tex` (old vs new), `paragraph.tex`.
* `api_usage.json`, `metadata.json`.
