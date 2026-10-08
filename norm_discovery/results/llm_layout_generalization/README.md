# Can GPT-5.5 propose layout-independent norms instead of layout-specific rules? (aisle domain)

Code: `experiments/llm_layout_generalization.py` (stages `prepare` -> `run` -> `analyze`).
* 20 new GPT-5.5 calls: 10 each for c2 and c3. All returned `gpt-5.5-2026-04-23`; no errors; 91.6k input and
  122.7k output tokens.
* Condition c1 reuses 10 cached GPT-5.5 responses (no new calls).
* No existing result was modified.

## Design
* **Layouts** (same norm `FORBIDDEN hasCart ∧ inAisle`, `inAisle := (shelfW∧shelfE) ∨ (shelfN∧shelfS)`; built with
  the existing `Layout` / `ShoppingDomain`; full grids in `setup.json`):
  * **L1:** the existing Exp. 2 store, with vertical aisles at columns 3/6/9, rows 0–2.
  * **L2:** a second store with a horizontal aisle (row 3, columns 3–8). Training data for c3 only.
  * **L3:** a test store never shown to the LLM. It has a vertical aisle (column 1) and a horizontal aisle (row 8).
    Cells that were aisle cells in L1/L2 are ordinary free cells here.
* **Evaluation sets:**
  * L1: the existing 10 held-out trajectories. The 5 involving aisle A3, which is never used in training, form the
    "unseen-aisle" subset.
  * L3: 10 diagnostic trajectories with verified labels, 6 compliant and 4 violating. They cover each aisle, compliant
    and violating, a cart parked inside an aisle, a standalone display, and two compliant **trap** paths that push the
    cart through cells that were aisles in L1 or L2.
* **Spurious, layout-specific competitors.** These are added to the Bayesian candidate set in every condition and
  are consistent with all training demonstrations of that condition (zero violations, checked):
  * cart forbidden in the cells demonstrators visited without it;
  * cart forbidden in the training layouts' aisle *coordinates*.
* **Conditions.** All use 10 independent GPT-5.5 calls (replicate keys `genlayout/<c>/<r>`). Run r uses the same
  initial (α₀) hypothesis list in every condition.
  * **c1, single layout, no guidance:** the existing refinement prompt and responses (cached runs `orig/aisle/r`).
  * **c2, single layout + guidance:** c1's exact prompt with one paragraph inserted (below). The paragraph never
    mentions carts, aisles, or "narrow".
  * **c3, two layouts + guidance:** the c2 prompt with the demonstration block replaced by L1 D1–D3 and L2 B1–B3
    (labelled "store A / store B").
  * Held constant: model, API-default generation settings, prompt template, vocabulary, output schema, 8 hypotheses
    requested, the current-hypotheses list, priors, likelihood, and the evaluation sets.
  * Varies: guidance (c1→c2); a second layout's demonstrations (c2→c3).
* **Guidance paragraph:** see `setup.json` / `GUIDANCE` in the script.
* **Outcomes:**
  * Proposal success: some valid GPT proposal is behaviourally equivalent to the intended norm on the probe sets of
    **all three** layouts. "L1-only" is equivalence on L1 alone.
  * Selection success: the Bayesian MAP over (initial ∪ refined ∪ spurious) hypotheses, given the condition's
    training demonstrations, is all-layout equivalent.
  * Generalisation success: the MAP classifies all 10 L3 diagnostics correctly.
  * Proposal kind: from the primitive atoms used. *Relational* = geometry only; *coordinate* = at(x,y) only; *mixed*
    = both; *other* = neither.

## Results (10 runs per condition; means [95% bootstrap CI])

| | c1: single layout | c2: + guidance | c3: two layouts + guidance |
|---|---|---|---|
| Proposals that are relational | 0.75 [0.66, 0.82] | 0.67 [0.63, 0.71] | 0.64 [0.60, 0.68] |
| Proposals using coordinates (coordinate or mixed) | 0.17 [0.07, 0.29] | **0.00** | **0.00** |
| Proposal success (all-layout equivalent) | **0/10** | **6/10** | **6/10** |
| Proposal success, L1-equivalent only | 9/10 | 9/10 | 6/10 |
| Selection success | 0/10 | 0/10 | **5/10** |
| MAP is a spurious coordinate rule | 9/10 | 10/10 | **0/10** |
| P(intended-equivalent \| D) | 0.00 | 0.08 [0.02, 0.14] | 0.25 [0.11, 0.39] |
| L1 held-out acc. / F1 | 0.73 / 0.61 | 0.70 / 0.57 | **1.00 / 1.00** |
| L1 unseen-aisle acc. | 0.46 | 0.40 | **1.00** |
| L3 (new layout) acc. / F1 | 0.52 / 0.06 | 0.50 / 0.00 | **0.90 [0.84, 0.96] / 0.92** |
| L3 acc.: vertical / horizontal / trap | 0.40 / 0.33 / 1.00 | 0.33 / 0.33 / 1.00 | 1.00 / 1.00 / 0.50 |
| Generalisation success (all L3 correct) | 0/10 | 0/10 | 5/10 |

## Interpretation

* **Guidance changes what GPT-5.5 proposes.** Coordinate-based proposals disappear (17% → 0%). Proposal of a truly
  layout-independent norm rises from 0/10 to 6/10.
  * Without guidance, GPT-5.5's relational proposals are usually "between shelves on the east and west", which is
    equivalent to the intended norm on L1 (9/10) but fails on horizontal aisles.
  * With guidance, it more often proposes orientation-independent definitions.
* **Guidance alone does not change selection.** With one layout, a cheap coordinate rule ("no cart in the 4 cells
  the demonstrators avoided") explains the demonstrations equally well and has a shorter description, so the Bayesian
  learner selects it in 9–10 of 10 runs. Held-out performance on L3 stays at chance (0.50–0.52, F1 ≈ 0).
  Proposal success ≠ selection success.
* **Two training layouts make selection and generalisation possible.** In c3 no spurious rule is ever selected. The
  relational norm explains both stores, while no coordinate rule consistent with both can match it. Selection
  succeeds in 5/10 runs, L1 accuracy is 1.00 (unseen aisle 1.00), and L3 accuracy is 0.90, F1 0.92.
  * The remaining c3 errors are *relational but non-equivalent* MAP hypotheses: "single-file next to a shelf"
    (adjShelf ∧ freeWidth ≤ 1). These correctly flag both L3 aisles, but also flag compliant trap paths along shelf
    edges and beside a display. Generalisation success = 5/10.
* **Caveats:**
  * c3 adds both a second layout *and* three more demonstrations, so layout diversity is not separated from data
    quantity.
  * L2 contains a horizontal aisle, so orientation is informed by c3's data but not by c1/c2's.
  * The current-hypotheses list in every prompt contains GPT-5.5's own earlier α₀ proposals. In some runs one of
    these is named, e.g., `no_cart_in_item_aisle`. It is identical across conditions, but it is a hint the original
    pipeline already gives.
  * 10 calls per condition gives coarse rates.
  * Equivalence is judged on finite probe sets (one per layout).

## Files
* `setup.json`: layouts, training and diagnostic trajectories, spurious rules, guidance text, and what is held
  constant or varied.
* `prompts.jsonl`: the exact c1/c2/c3 prompts.
* `llm/<c>/run_XX/transcript.json`: prompt, raw response, parsed hypotheses, model, request id, usage.
* `runs.jsonl` / `runs.csv`: per run. `proposals.csv`: per proposal (kind, equivalence, accuracies, GPT's
  justification). `summary.csv`.
* `table.tex`, `paragraph.tex`.
* `api_usage.json`, `metadata.json`.
