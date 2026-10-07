# GPT-5.5 original-protocol reproduction (3 demonstrations, rho = 0.1)

Mean ± SD over independent GPT-5.5 runs (MLCI is deterministic and uses no LLM).

| Method | Exp.1 Acc. | Exp.1 Viol. F1 | Exp.2 Acc. | Exp.2 Viol. F1 | Exp.2 Unseen Acc. |
|---|---|---|---|---|---|
| Fixed abstraction (GPT-5.5 initial proposals) | 0.80 ± 0.00 | 0.80 ± 0.00 | 0.60 ± 0.00 | 0.33 ± 0.00 | 0.40 ± 0.00 |
| MLCI (no LLM) | 0.60 | 0.75 | 0.50 | 0.00 | 0.40 |
| Ours – GPT-5.5 | 0.80 ± 0.00 | 0.80 ± 0.00 | 0.60 ± 0.00 | 0.33 ± 0.00 | 0.40 ± 0.00 |
| Oracle abstraction (GPT-5.5 proposals) | 0.96 ± 0.08 | 0.96 ± 0.08 | 0.98 ± 0.06 | 0.98 ± 0.05 | 1.00 ± 0.00 |

| Domain | Method | runs (errors) | refined | refinement at t | intended proposed | final MAP = intended | mean P(intended) | accepted / rejected proposals (mean) |
|---|---|---|---|---|---|---|---|---|
| cart | full | 10 (0) | 0/10 | [] | 0/10 | 0/10 | 0.000 | 8.0 / 0.0 |
| cart | fixed | 10 (0) | 0/10 | [] | 0/10 | 0/10 | 0.000 | 8.0 / 0.0 |
| cart | oracle | 10 (0) | 0/10 | [] | 7/10 | 5/10 | 0.562 | 16.0 / 0.0 |
| aisle | full | 10 (0) | 0/10 | [] | 0/10 | 0/10 | 0.000 | 8.0 / 0.0 |
| aisle | fixed | 10 (0) | 0/10 | [] | 0/10 | 0/10 | 0.000 | 8.0 / 0.0 |
| aisle | oracle | 10 (0) | 0/10 | [] | 10/10 | 7/10 | 0.471 | 15.9 / 0.1 |

Paper (Table I): GPT-5.5 Exp.1 0.80/0.80, Exp.2 0.96/1.00/0.93; Table II: GPT-5.5 proposed 10/10 and 10/10, final MAP 0/10 and 8/10.

## Refinement trigger per run (full method)

| Domain | run | log p2 | log p3 | log(ρ·b3) | p3/b3 | triggered |
|---|---|---|---|---|---|---|
| cart | 0 | -26.94 | -23.36 | -29.24 | 35.8 | False |
| cart | 1 | -26.97 | -23.36 | -29.27 | 36.9 | False |
| cart | 2 | -26.94 | -23.36 | -29.24 | 35.9 | False |
| cart | 3 | -26.94 | -23.36 | -29.24 | 35.9 | False |
| cart | 4 | -26.94 | -23.36 | -29.24 | 35.8 | False |
| cart | 5 | -26.94 | -23.36 | -29.24 | 35.8 | False |
| cart | 6 | -26.94 | -23.36 | -29.24 | 35.9 | False |
| cart | 7 | -26.94 | -23.36 | -29.24 | 35.9 | False |
| cart | 8 | -26.94 | -23.36 | -29.24 | 35.9 | False |
| cart | 9 | -26.94 | -23.36 | -29.24 | 35.8 | False |
| aisle | 0 | -14.56 | -5.03 | -16.86 | 1.37e+04 | False |
| aisle | 1 | -14.56 | -5.03 | -16.86 | 1.37e+04 | False |
| aisle | 2 | -14.56 | -5.03 | -16.86 | 1.37e+04 | False |
| aisle | 3 | -14.56 | -5.03 | -16.86 | 1.37e+04 | False |
| aisle | 4 | -14.57 | -5.03 | -16.87 | 1.38e+04 | False |
| aisle | 5 | -14.56 | -5.03 | -16.86 | 1.37e+04 | False |
| aisle | 6 | -14.56 | -5.03 | -16.86 | 1.37e+04 | False |
| aisle | 7 | -14.56 | -5.03 | -16.87 | 1.38e+04 | False |
| aisle | 8 | -14.57 | -5.03 | -16.87 | 1.38e+04 | False |
| aisle | 9 | -14.56 | -5.03 | -16.86 | 1.37e+04 | False |
