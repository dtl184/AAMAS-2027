| Sequence | t | Demo context | P(H_int) | P(alt) | P(others) | Entropy | log BF int:alt | MAP | Acc. |
|---|---|---|---|---|---|---|---|---|---|
| diagnostic | 0 | (prior) | 0.016 | 0.115 | 0.869 | 2.16 | 0.00 | H_null | 0.40 |
| diagnostic | 1 | apple+milk | cart=station | 0.112 | 0.725 | 0.163 | 0.77 | 0.13 | A0_any_to_last | 0.80 |
| diagnostic | 2 | bread+eggs | cart=station | 0.131 | 0.744 | 0.124 | 0.75 | 0.26 | A0_any_to_last | 0.80 |
| diagnostic | 3 | apple+bread+milk | cart=station | 0.152 | 0.754 | 0.093 | 0.72 | 0.40 | A0_any_to_last | 0.80 |
| diagnostic | 4 | eggs+milk | cart=station | 0.174 | 0.756 | 0.069 | 0.70 | 0.53 | A0_any_to_last | 0.80 |
| diagnostic | 5 | apple+bread | cart=station | 0.198 | 0.751 | 0.051 | 0.69 | 0.67 | A0_any_to_last | 0.80 |
| diagnostic | 6 | bread | cart=(3, 1) | 1.000 | 0.000 | 0.000 | 0.00 | 18.85 | H_int | 1.00 |
| diagnostic | 7 | bread+milk | cart=station | 1.000 | 0.000 | 0.000 | 0.00 | 19.05 | H_int | 1.00 |
| diagnostic | 8 | apple+eggs+milk | cart=station | 1.000 | 0.000 | 0.000 | 0.00 | 19.44 | H_int | 1.00 |
| control | 0 | (prior) | 0.016 | 0.115 | 0.869 | 2.16 | 0.00 | H_null | 0.40 |
| control | 1 | apple+milk | cart=station | 0.112 | 0.725 | 0.163 | 0.77 | 0.13 | A0_any_to_last | 0.80 |
| control | 2 | bread+eggs | cart=station | 0.131 | 0.744 | 0.124 | 0.75 | 0.26 | A0_any_to_last | 0.80 |
| control | 3 | apple+bread+milk | cart=station | 0.152 | 0.754 | 0.093 | 0.72 | 0.40 | A0_any_to_last | 0.80 |
| control | 4 | eggs+milk | cart=station | 0.174 | 0.756 | 0.069 | 0.70 | 0.53 | A0_any_to_last | 0.80 |
| control | 5 | apple+bread | cart=station | 0.198 | 0.751 | 0.051 | 0.69 | 0.67 | A0_any_to_last | 0.80 |
| control | 6 | bread+milk | cart=station | 0.223 | 0.740 | 0.037 | 0.68 | 0.80 | A0_any_to_last | 0.80 |
| control | 7 | apple+eggs+milk | cart=station | 0.250 | 0.723 | 0.027 | 0.68 | 0.94 | A0_any_to_last | 0.80 |
| control | 8 | apple+eggs | cart=station | 0.277 | 0.703 | 0.019 | 0.68 | 1.07 | A0_any_to_last | 0.80 |
| reacquire | 0 | (prior) | 0.016 | 0.115 | 0.869 | 2.16 | 0.00 | H_null | 0.40 |
| reacquire | 1 | apple+milk | cart=station | 0.112 | 0.725 | 0.163 | 0.77 | 0.13 | A0_any_to_last | 0.80 |
| reacquire | 2 | bread+eggs | cart=station | 0.131 | 0.744 | 0.124 | 0.75 | 0.26 | A0_any_to_last | 0.80 |
| reacquire | 3 | apple+bread+milk | cart=station | 0.152 | 0.754 | 0.093 | 0.72 | 0.40 | A0_any_to_last | 0.80 |
| reacquire | 4 | eggs+milk | cart=station | 0.174 | 0.756 | 0.069 | 0.70 | 0.53 | A0_any_to_last | 0.80 |
| reacquire | 5 | apple+bread | cart=station | 0.198 | 0.751 | 0.051 | 0.69 | 0.67 | A0_any_to_last | 0.80 |
| reacquire | 6 | apple+milk | cart=station | 0.223 | 0.740 | 0.037 | 0.68 | 0.80 | A0_any_to_last | 0.80 |
| reacquire | 7 | bread+milk | cart=station | 0.250 | 0.723 | 0.027 | 0.68 | 0.94 | A0_any_to_last | 0.80 |
| reacquire | 8 | apple+eggs+milk | cart=station | 0.277 | 0.703 | 0.019 | 0.68 | 1.07 | A0_any_to_last | 0.80 |

| Extra demo after D1–D5 | n | P(H_int) [95% CI] | log odds int:alt | Acc. | fraction MAP=H_int |
|---|---|---|---|---|---|
| active | 1 | 1.000 [1.000, 1.000] | 16.85 | 1.00 | 1.00 |
| random_any | 50 | 0.317 [0.257, 0.391] | -0.04 | 0.83 | 0.14 |
| random_standard | 50 | 0.223 [0.223, 0.223] | -1.20 | 0.80 | 0.00 |

| Competitor | P after D1–D5 | Most diagnostic context | MI (nats, equal pair prior) | E|log BF| | Best ordinary context MI |
|---|---|---|---|---|---|
| A0_interact_last | 0.000 | apple+milk | cart=parked(0, 0) | 0.693 | 17.2 | 0.692 |
| A0_visit_last | 0.000 | apple+eggs | cart=parked(4, 0) | 0.693 | 17.2 | 0.693 |
| A0_any_to_last | 0.751 | apple+bread | cart=parked(4, 2) | 0.693 | 19.3 | 0.046 |
| A0_first_to_last | 0.000 | apple+milk | cart=parked(8, 4) | 0.693 | 17.2 | 0.692 |
| H_uncond | 0.000 | bread+milk | cart=parked(0, 1) | 0.693 | 17.2 | 0.692 |
| H_item | 0.051 | apple+bread | cart=parked(4, 2) | 0.693 | 23.1 | 0.141 |
| H_visit | 0.000 | apple+milk | cart=parked(1, 0) | 0.693 | 17.2 | 0.692 |
| H_leave | 0.000 | apple+bread | cart=parked(9, 5) | 0.693 | 17.2 | 0.693 |
| H_noexit | 0.000 | apple+bread | cart=parked(9, 5) | 0.693 | 17.2 | 0.693 |
