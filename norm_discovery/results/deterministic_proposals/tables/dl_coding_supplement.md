| Coding | Domain | Protocol | Method | ρ / baseline | Acc. | Unseen acc. | P(intended) | MAP counts |
|---|---|---|---|---|---|---|---|---|
| nodes | cart | original | fixed | – / min | 0.80 [0.80, 0.80] | – | 0.000 | {"A0_any_to_last": 1} |
| nodes | cart | original | full | 0.1 / min | 0.80 [0.80, 0.80] | – | 0.000 | {"A0_any_to_last": 1} |
| nodes | cart | original | full | 0.4 / min | 0.80 [0.80, 0.80] | – | 0.000 | {"A0_any_to_last": 1} |
| nodes | cart | original | full | 0.8 / median | 0.80 [0.80, 0.80] | – | 0.152 | {"A0_any_to_last": 1} |
| nodes | cart | original | oracle | – / min | 0.80 [0.80, 0.80] | – | 0.152 | {"A0_any_to_last": 1} |
| nodes | cart | pool | fixed | – / min | 0.80 [0.80, 0.80] | – | 0.000 | {"A0_any_to_last": 20} |
| nodes | cart | pool | full | 0.1 / min | 0.81 [0.80, 0.83] | – | 0.033 | {"A0_any_to_last": 19, "H_int": 1} |
| nodes | cart | pool | full | 0.4 / min | 0.85 [0.81, 0.89] | – | 0.166 | {"A0_any_to_last": 15, "H_int": 5} |
| nodes | cart | pool | full | 0.8 / median | 1.00 [1.00, 1.00] | – | 0.664 | {"H_int": 20} |
| nodes | cart | pool | oracle | – / min | 1.00 [1.00, 1.00] | – | 0.664 | {"H_int": 20} |
| nodes | aisle | original | fixed | – / min | 0.60 [0.60, 0.60] | 0.40 | 0.000 | {"B0_cells": 1} |
| nodes | aisle | original | full | 0.1 / min | 0.60 [0.60, 0.60] | 0.40 | 0.000 | {"B0_cells": 1} |
| nodes | aisle | original | full | 0.4 / min | 0.60 [0.60, 0.60] | 0.40 | 0.000 | {"B0_cells": 1} |
| nodes | aisle | original | full | 0.8 / median | 0.60 [0.60, 0.60] | 0.40 | 0.000 | {"B0_cells": 1} |
| nodes | aisle | original | oracle | – / min | 1.00 [1.00, 1.00] | 1.00 | 0.382 | {"S_corridor": 1} |
| nodes | aisle | pool | fixed | – / min | 0.55 [0.53, 0.58] | 0.40 | 0.000 | {"B0_cells": 10, "H_null": 10} |
| nodes | aisle | pool | full | 0.1 / min | 0.62 [0.58, 0.66] | 0.40 | 0.131 | {"B0_cells": 4, "S_memorised": 10, "H_null": 6} |
| nodes | aisle | pool | full | 0.4 / min | 0.68 [0.66, 0.70] | 0.40 | 0.222 | {"S_memorised": 17, "B0_cells": 2, "H_null": 1} |
| nodes | aisle | pool | full | 0.8 / median | 0.70 [0.70, 0.70] | 0.40 | 0.262 | {"S_memorised": 19, "B0_cells": 1} |
| nodes | aisle | pool | oracle | – / min | 0.98 [0.95, 1.00] | 0.97 | 0.475 | {"S_corridor": 19, "B0_cells": 1} |
| tokens | cart | original | fixed | – / min | 0.80 [0.80, 0.80] | – | 0.000 | {"A0_any_to_last": 1} |
| tokens | cart | original | full | 0.1 / min | 0.80 [0.80, 0.80] | – | 0.000 | {"A0_any_to_last": 1} |
| tokens | cart | original | full | 0.4 / min | 0.80 [0.80, 0.80] | – | 0.000 | {"A0_any_to_last": 1} |
| tokens | cart | original | full | 0.8 / median | 0.80 [0.80, 0.80] | – | 0.152 | {"A0_any_to_last": 1} |
| tokens | cart | original | oracle | – / min | 0.80 [0.80, 0.80] | – | 0.152 | {"A0_any_to_last": 1} |
| tokens | cart | pool | fixed | – / min | 0.80 [0.80, 0.80] | – | 0.000 | {"A0_any_to_last": 20} |
| tokens | cart | pool | full | 0.1 / min | 0.81 [0.80, 0.83] | – | 0.033 | {"A0_any_to_last": 19, "H_int": 1} |
| tokens | cart | pool | full | 0.4 / min | 0.85 [0.81, 0.89] | – | 0.166 | {"A0_any_to_last": 15, "H_int": 5} |
| tokens | cart | pool | full | 0.8 / median | 1.00 [1.00, 1.00] | – | 0.664 | {"H_int": 20} |
| tokens | cart | pool | oracle | – / min | 1.00 [1.00, 1.00] | – | 0.664 | {"H_int": 20} |
| tokens | aisle | original | fixed | – / min | 0.60 [0.60, 0.60] | 0.40 | 0.000 | {"B0_pickcells": 1} |
| tokens | aisle | original | full | 0.1 / min | 0.60 [0.60, 0.60] | 0.40 | 0.000 | {"B0_pickcells": 1} |
| tokens | aisle | original | full | 0.4 / min | 0.60 [0.60, 0.60] | 0.40 | 0.000 | {"B0_pickcells": 1} |
| tokens | aisle | original | full | 0.8 / median | 0.60 [0.60, 0.60] | 0.40 | 0.000 | {"B0_pickcells": 1} |
| tokens | aisle | original | oracle | – / min | 1.00 [1.00, 1.00] | 1.00 | 0.389 | {"S_corridor": 1} |
| tokens | aisle | pool | fixed | – / min | 0.55 [0.53, 0.58] | 0.40 | 0.000 | {"B0_cells": 10, "H_null": 10} |
| tokens | aisle | pool | full | 0.1 / min | 0.78 [0.68, 0.88] | 0.70 | 0.209 | {"B0_cells": 4, "S_corridor": 10, "H_null": 6} |
| tokens | aisle | pool | full | 0.4 / min | 0.94 [0.87, 1.00] | 0.91 | 0.351 | {"S_corridor": 17, "B0_cells": 2, "H_null": 1} |
| tokens | aisle | pool | full | 0.8 / median | 1.00 [1.00, 1.00] | 1.00 | 0.415 | {"S_corridor": 20} |
| tokens | aisle | pool | oracle | – / min | 1.00 [1.00, 1.00] | 1.00 | 0.483 | {"S_corridor": 20} |
