## original_protocol

| Domain | Method | Acc. | Viol. F1 | Unseen acc. | MAP hypothesis | P(intended) | Refinements |
|---|---|---|---|---|---|---|---|
| cart | fixed | 0.80 | 0.80 | – | A0_any_to_last | – | 0 |
| cart | full | 0.80 | 0.80 | – | A0_any_to_last | – | 0 |
| cart | oracle | 0.80 | 0.80 | – | A0_any_to_last | 0.152 | 0 |
| cart | mlci | 0.60 | 0.75 | – | – | – | – |
| aisle | fixed | 0.60 | 0.33 | 0.40 | B0_cells | – | 0 |
| aisle | full | 0.60 | 0.33 | 0.40 | B0_cells | – | 0 |
| aisle | oracle | 1.00 | 1.00 | 1.00 | S_corridor | 0.382 | 0 |
| aisle | mlci | 0.50 | 0.00 | 0.40 | – | – | – |

## threshold_sensitivity

| Domain | Protocol | Baseline | ρ | Acc. [95% CI] | Viol. F1 | Unseen acc. | Refinement rate | Mean #ref. | P(intended) |
|---|---|---|---|---|---|---|---|---|---|
| aisle | original | median | 0.01 | 0.60 [0.53, 0.67] | 0.30 | 0.40 | 0.33 | 0.33 | 0.075 |
| aisle | original | min | 0.01 | 0.60 [0.53, 0.67] | 0.30 | 0.40 | 0.33 | 0.33 | 0.075 |
| aisle | original | median | 0.025 | 0.60 [0.53, 0.67] | 0.30 | 0.40 | 0.33 | 0.33 | 0.075 |
| aisle | original | min | 0.025 | 0.60 [0.53, 0.67] | 0.30 | 0.40 | 0.33 | 0.33 | 0.075 |
| aisle | original | median | 0.05 | 0.60 [0.53, 0.67] | 0.30 | 0.40 | 0.33 | 0.33 | 0.075 |
| aisle | original | min | 0.05 | 0.60 [0.53, 0.67] | 0.30 | 0.40 | 0.33 | 0.33 | 0.075 |
| aisle | original | median | 0.1 **(paper)** | 0.60 [0.53, 0.67] | 0.30 | 0.40 | 0.33 | 0.33 | 0.075 |
| aisle | original | min | 0.1 **(paper)** | 0.60 [0.53, 0.67] | 0.30 | 0.40 | 0.33 | 0.33 | 0.075 |
| aisle | original | median | 0.2 | 0.60 [0.53, 0.67] | 0.30 | 0.40 | 0.33 | 0.33 | 0.075 |
| aisle | original | min | 0.2 | 0.60 [0.53, 0.67] | 0.30 | 0.40 | 0.33 | 0.33 | 0.075 |
| aisle | original | median | 0.4 | 0.60 [0.53, 0.67] | 0.30 | 0.40 | 0.33 | 0.33 | 0.075 |
| aisle | original | min | 0.4 | 0.60 [0.53, 0.67] | 0.30 | 0.40 | 0.33 | 0.33 | 0.075 |
| aisle | original | median | 0.6 | 0.63 [0.57, 0.68] | 0.40 | 0.40 | 0.50 | 0.50 | 0.114 |
| aisle | original | min | 0.6 | 0.63 [0.57, 0.68] | 0.40 | 0.40 | 0.50 | 0.50 | 0.114 |
| aisle | original | median | 0.8 | 0.63 [0.57, 0.68] | 0.40 | 0.40 | 0.50 | 0.50 | 0.114 |
| aisle | original | min | 0.8 | 0.63 [0.57, 0.68] | 0.40 | 0.40 | 0.50 | 0.50 | 0.114 |
| aisle | pool | median | 0.01 | 0.68 [0.65, 0.73] | 0.52 | 0.43 | 0.75 | 6.20 | 0.207 |
| aisle | pool | min | 0.01 | 0.61 [0.57, 0.64] | 0.33 | 0.40 | 0.35 | 2.20 | 0.092 |
| aisle | pool | median | 0.025 | 0.68 [0.65, 0.73] | 0.52 | 0.43 | 0.75 | 7.70 | 0.207 |
| aisle | pool | min | 0.025 | 0.61 [0.57, 0.65] | 0.34 | 0.40 | 0.40 | 3.80 | 0.105 |
| aisle | pool | median | 0.05 | 0.69 [0.66, 0.74] | 0.54 | 0.43 | 0.85 | 8.20 | 0.233 |
| aisle | pool | min | 0.05 | 0.62 [0.58, 0.65] | 0.35 | 0.40 | 0.45 | 4.15 | 0.118 |
| aisle | pool | median | 0.1 **(paper)** | 0.69 [0.68, 0.70] | 0.56 | 0.40 | 0.95 | 8.40 | 0.248 |
| aisle | pool | min | 0.1 **(paper)** | 0.62 [0.58, 0.66] | 0.36 | 0.40 | 0.50 | 4.50 | 0.131 |
| aisle | pool | median | 0.2 | 0.70 [0.70, 0.70] | 0.57 | 0.40 | 1.00 | 9.00 | 0.262 |
| aisle | pool | min | 0.2 | 0.68 [0.65, 0.70] | 0.52 | 0.40 | 0.80 | 4.80 | 0.208 |
| aisle | pool | median | 0.4 | 0.70 [0.70, 0.70] | 0.57 | 0.40 | 1.00 | 9.10 | 0.262 |
| aisle | pool | min | 0.4 | 0.68 [0.66, 0.70] | 0.53 | 0.40 | 0.85 | 4.85 | 0.222 |
| aisle | pool | median | 0.6 | 0.70 [0.70, 0.70] | 0.57 | 0.40 | 1.00 | 9.75 | 0.262 |
| aisle | pool | min | 0.6 | 0.69 [0.67, 0.70] | 0.54 | 0.40 | 0.95 | 5.05 | 0.248 |
| aisle | pool | median | 0.8 | 0.70 [0.70, 0.70] | 0.57 | 0.40 | 1.00 | 10.65 | 0.262 |
| aisle | pool | min | 0.8 | 0.69 [0.67, 0.70] | 0.54 | 0.40 | 0.95 | 5.55 | 0.248 |
| cart | original | median | 0.01 | 0.80 [0.80, 0.80] | 0.80 | – | 0.00 | 0.00 | 0.000 |
| cart | original | min | 0.01 | 0.80 [0.80, 0.80] | 0.80 | – | 0.00 | 0.00 | 0.000 |
| cart | original | median | 0.025 | 0.80 [0.80, 0.80] | 0.80 | – | 0.00 | 0.00 | 0.000 |
| cart | original | min | 0.025 | 0.80 [0.80, 0.80] | 0.80 | – | 0.00 | 0.00 | 0.000 |
| cart | original | median | 0.05 | 0.80 [0.80, 0.80] | 0.80 | – | 0.00 | 0.00 | 0.000 |
| cart | original | min | 0.05 | 0.80 [0.80, 0.80] | 0.80 | – | 0.00 | 0.00 | 0.000 |
| cart | original | median | 0.1 **(paper)** | 0.80 [0.80, 0.80] | 0.80 | – | 0.00 | 0.00 | 0.000 |
| cart | original | min | 0.1 **(paper)** | 0.80 [0.80, 0.80] | 0.80 | – | 0.00 | 0.00 | 0.000 |
| cart | original | median | 0.2 | 0.80 [0.80, 0.80] | 0.80 | – | 0.00 | 0.00 | 0.000 |
| cart | original | min | 0.2 | 0.80 [0.80, 0.80] | 0.80 | – | 0.00 | 0.00 | 0.000 |
| cart | original | median | 0.4 | 0.80 [0.80, 0.80] | 0.80 | – | 0.00 | 0.00 | 0.000 |
| cart | original | min | 0.4 | 0.80 [0.80, 0.80] | 0.80 | – | 0.00 | 0.00 | 0.000 |
| cart | original | median | 0.6 | 0.80 [0.80, 0.80] | 0.80 | – | 0.17 | 0.17 | 0.025 |
| cart | original | min | 0.6 | 0.80 [0.80, 0.80] | 0.80 | – | 0.17 | 0.17 | 0.025 |
| cart | original | median | 0.8 | 0.80 [0.80, 0.80] | 0.80 | – | 0.33 | 0.33 | 0.051 |
| cart | original | min | 0.8 | 0.80 [0.80, 0.80] | 0.80 | – | 0.33 | 0.33 | 0.051 |
| cart | pool | median | 0.01 | 0.80 [0.80, 0.80] | 0.80 | – | 0.00 | 0.00 | 0.000 |
| cart | pool | min | 0.01 | 0.80 [0.80, 0.80] | 0.80 | – | 0.00 | 0.00 | 0.000 |
| cart | pool | median | 0.025 | 0.81 [0.80, 0.83] | 0.81 | – | 0.05 | 0.15 | 0.033 |
| cart | pool | min | 0.025 | 0.81 [0.80, 0.83] | 0.81 | – | 0.05 | 0.15 | 0.033 |
| cart | pool | median | 0.05 | 0.81 [0.80, 0.83] | 0.81 | – | 0.05 | 0.30 | 0.033 |
| cart | pool | min | 0.05 | 0.81 [0.80, 0.83] | 0.81 | – | 0.05 | 0.30 | 0.033 |
| cart | pool | median | 0.1 **(paper)** | 0.84 [0.81, 0.88] | 0.84 | – | 0.20 | 0.85 | 0.133 |
| cart | pool | min | 0.1 **(paper)** | 0.81 [0.80, 0.83] | 0.81 | – | 0.05 | 0.30 | 0.033 |
| cart | pool | median | 0.2 | 0.91 [0.87, 0.95] | 0.91 | – | 0.55 | 4.05 | 0.365 |
| cart | pool | min | 0.2 | 0.84 [0.81, 0.88] | 0.84 | – | 0.20 | 1.10 | 0.133 |
| cart | pool | median | 0.4 | 0.99 [0.97, 1.00] | 0.99 | – | 0.95 | 8.10 | 0.631 |
| cart | pool | min | 0.4 | 0.85 [0.81, 0.89] | 0.85 | – | 0.25 | 2.25 | 0.166 |
| cart | pool | median | 0.6 | 0.99 [0.97, 1.00] | 0.99 | – | 0.95 | 11.00 | 0.631 |
| cart | pool | min | 0.6 | 0.92 [0.88, 0.96] | 0.92 | – | 0.60 | 4.55 | 0.398 |
| cart | pool | median | 0.8 | 1.00 [1.00, 1.00] | 1.00 | – | 1.00 | 12.50 | 0.664 |
| cart | pool | min | 0.8 | 0.94 [0.90, 0.98] | 0.94 | – | 0.70 | 6.35 | 0.465 |

Reference (ρ-independent) on the same sequences:

| Domain | Protocol | Method | Acc. [95% CI] | Unseen acc. | P(intended) |
|---|---|---|---|---|---|
| aisle | original | fixed | 0.57 [0.53, 0.60] | 0.40 | 0.000 |
| aisle | original | oracle | 1.00 [1.00, 1.00] | 1.00 | 0.388 |
| aisle | pool | fixed | 0.55 [0.53, 0.58] | 0.40 | 0.000 |
| aisle | pool | oracle | 0.98 [0.95, 1.00] | 0.97 | 0.475 |
| cart | original | fixed | 0.80 [0.80, 0.80] | – | 0.000 |
| cart | original | oracle | 0.80 [0.80, 0.80] | – | 0.152 |
| cart | pool | fixed | 0.80 [0.80, 0.80] | – | 0.000 |
| cart | pool | oracle | 1.00 [1.00, 1.00] | – | 0.664 |

## noise_robustness

| Domain | Noise | q | Method | Acc. [95% CI] | Viol. F1 [95% CI] | Unseen acc. | P(intended) | MAP=intended | MAP≡intended (held-out) | Refinements |
|---|---|---|---|---|---|---|---|---|---|---|
| cart | behavioural | 0.0 | full | 0.82 [0.80, 0.85] | 0.82 [0.80, 0.85] | – | 0.066 | 0.10 | 0.10 | 0.10 |
| cart | behavioural | 0.0 | fixed | 0.80 [0.80, 0.80] | 0.80 [0.80, 0.80] | – | 0.000 | 0.00 | 0.00 | 0.00 |
| cart | behavioural | 0.0 | oracle | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | – | 0.664 | 1.00 | 1.00 | 0.00 |
| cart | behavioural | 0.0 | mlci | 0.79 [0.72, 0.86] | 0.74 [0.64, 0.84] | – | – | – | – | – |
| cart | behavioural | 0.05 | full | 0.99 [0.97, 1.00] | 0.99 [0.97, 1.00] | – | 0.631 | 0.95 | 0.95 | 1.05 |
| cart | behavioural | 0.05 | fixed | 0.80 [0.80, 0.80] | 0.80 [0.80, 0.80] | – | 0.000 | 0.00 | 0.00 | 0.00 |
| cart | behavioural | 0.05 | oracle | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | – | 0.664 | 1.00 | 1.00 | 0.00 |
| cart | behavioural | 0.05 | mlci | 0.80 [0.74, 0.85] | 0.78 [0.69, 0.85] | – | – | – | – | – |
| cart | behavioural | 0.1 | full | 0.99 [0.97, 1.00] | 0.99 [0.97, 1.00] | – | 0.631 | 0.95 | 0.95 | 1.90 |
| cart | behavioural | 0.1 | fixed | 0.80 [0.80, 0.80] | 0.80 [0.80, 0.80] | – | 0.000 | 0.00 | 0.00 | 0.00 |
| cart | behavioural | 0.1 | oracle | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | – | 0.664 | 1.00 | 1.00 | 0.00 |
| cart | behavioural | 0.1 | mlci | 0.82 [0.77, 0.86] | 0.81 [0.73, 0.86] | – | – | – | – | – |
| cart | behavioural | 0.2 | full | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | – | 0.664 | 1.00 | 1.00 | 3.45 |
| cart | behavioural | 0.2 | fixed | 0.80 [0.80, 0.80] | 0.80 [0.80, 0.80] | – | 0.000 | 0.00 | 0.00 | 0.00 |
| cart | behavioural | 0.2 | oracle | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | – | 0.664 | 1.00 | 1.00 | 0.00 |
| cart | behavioural | 0.2 | mlci | 0.83 [0.80, 0.86] | 0.83 [0.78, 0.86] | – | – | – | – | – |
| cart | behavioural | 0.3 | full | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | – | 0.664 | 1.00 | 1.00 | 4.85 |
| cart | behavioural | 0.3 | fixed | 0.80 [0.80, 0.80] | 0.80 [0.80, 0.80] | – | 0.000 | 0.00 | 0.00 | 0.00 |
| cart | behavioural | 0.3 | oracle | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | – | 0.664 | 1.00 | 1.00 | 0.00 |
| cart | behavioural | 0.3 | mlci | 0.82 [0.77, 0.86] | 0.81 [0.73, 0.86] | – | – | – | – | – |
| cart | violation | 0.0 | full | 0.82 [0.80, 0.85] | 0.82 [0.80, 0.85] | – | 0.066 | 0.10 | 0.10 | 0.10 |
| cart | violation | 0.0 | fixed | 0.80 [0.80, 0.80] | 0.80 [0.80, 0.80] | – | 0.000 | 0.00 | 0.00 | 0.00 |
| cart | violation | 0.0 | oracle | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | – | 0.664 | 1.00 | 1.00 | 0.00 |
| cart | violation | 0.0 | mlci | 0.79 [0.72, 0.86] | 0.74 [0.64, 0.84] | – | – | – | – | – |
| cart | violation | 0.05 | full | 0.84 [0.81, 0.88] | 0.84 [0.81, 0.88] | – | 0.127 | 0.20 | 0.20 | 0.30 |
| cart | violation | 0.05 | fixed | 0.80 [0.80, 0.80] | 0.80 [0.80, 0.80] | – | 0.000 | 0.00 | 0.00 | 0.00 |
| cart | violation | 0.05 | oracle | 0.92 [0.88, 0.96] | 0.92 [0.88, 0.96] | – | 0.381 | 0.60 | 0.60 | 0.00 |
| cart | violation | 0.05 | mlci | 0.74 [0.66, 0.80] | 0.67 [0.54, 0.78] | – | – | – | – | – |
| cart | violation | 0.1 | full | 0.82 [0.80, 0.85] | 0.82 [0.80, 0.85] | – | 0.060 | 0.10 | 0.10 | 0.35 |
| cart | violation | 0.1 | fixed | 0.80 [0.80, 0.80] | 0.80 [0.80, 0.80] | – | 0.000 | 0.00 | 0.00 | 0.00 |
| cart | violation | 0.1 | oracle | 0.87 [0.83, 0.91] | 0.87 [0.83, 0.91] | – | 0.211 | 0.35 | 0.35 | 0.00 |
| cart | violation | 0.1 | mlci | 0.62 [0.54, 0.70] | 0.46 [0.31, 0.60] | – | – | – | – | – |
| cart | violation | 0.2 | full | 0.78 [0.74, 0.80] | 0.76 [0.68, 0.80] | – | 0.000 | 0.00 | 0.00 | 1.50 |
| cart | violation | 0.2 | fixed | 0.78 [0.74, 0.80] | 0.76 [0.68, 0.80] | – | 0.000 | 0.00 | 0.00 | 0.00 |
| cart | violation | 0.2 | oracle | 0.82 [0.80, 0.85] | 0.82 [0.80, 0.85] | – | 0.054 | 0.10 | 0.10 | 0.00 |
| cart | violation | 0.2 | mlci | 0.49 [0.44, 0.57] | 0.21 [0.08, 0.34] | – | – | – | – | – |
| cart | violation | 0.3 | full | 0.75 [0.69, 0.80] | 0.71 [0.59, 0.80] | – | 0.000 | 0.00 | 0.00 | 1.35 |
| cart | violation | 0.3 | fixed | 0.75 [0.69, 0.80] | 0.71 [0.59, 0.80] | – | 0.000 | 0.00 | 0.00 | 0.00 |
| cart | violation | 0.3 | oracle | 0.77 [0.72, 0.80] | 0.75 [0.65, 0.80] | – | 0.000 | 0.00 | 0.00 | 0.00 |
| cart | violation | 0.3 | mlci | 0.42 [0.40, 0.47] | 0.05 [0.00, 0.15] | – | – | – | – | – |
| aisle | behavioural | 0.0 | full | 0.62 [0.58, 0.65] | 0.36 [0.25, 0.45] | 0.40 | 0.092 | 0.00 | 0.00 | 1.95 |
| aisle | behavioural | 0.0 | fixed | 0.58 [0.55, 0.60] | 0.26 [0.18, 0.34] | 0.40 | 0.000 | 0.00 | 0.00 | 0.00 |
| aisle | behavioural | 0.0 | oracle | 0.92 [0.84, 0.98] | 0.92 [0.85, 0.98] | 0.90 | 0.422 | 0.00 | 0.80 | 0.00 |
| aisle | behavioural | 0.0 | mlci | 0.59 [0.58, 0.60] | 0.50 [0.49, 0.50] | 0.40 | – | – | – | – |
| aisle | behavioural | 0.05 | full | 0.70 [0.70, 0.70] | 0.57 [0.57, 0.57] | 0.40 | 0.263 | 0.00 | 0.00 | 2.85 |
| aisle | behavioural | 0.05 | fixed | 0.58 [0.55, 0.60] | 0.26 [0.18, 0.34] | 0.40 | 0.000 | 0.00 | 0.00 | 0.00 |
| aisle | behavioural | 0.05 | oracle | 0.97 [0.93, 1.00] | 0.96 [0.89, 1.00] | 0.94 | 0.467 | 0.00 | 0.90 | 0.00 |
| aisle | behavioural | 0.05 | mlci | 0.69 [0.68, 0.70] | 0.57 [0.56, 0.57] | 0.40 | – | – | – | – |
| aisle | behavioural | 0.1 | full | 0.70 [0.70, 0.70] | 0.57 [0.57, 0.57] | 0.40 | 0.249 | 0.00 | 0.00 | 3.55 |
| aisle | behavioural | 0.1 | fixed | 0.58 [0.55, 0.60] | 0.26 [0.18, 0.34] | 0.40 | 0.000 | 0.00 | 0.00 | 0.00 |
| aisle | behavioural | 0.1 | oracle | 0.97 [0.93, 1.00] | 0.96 [0.89, 1.00] | 0.94 | 0.473 | 0.00 | 0.90 | 0.00 |
| aisle | behavioural | 0.1 | mlci | 0.69 [0.68, 0.70] | 0.57 [0.56, 0.57] | 0.40 | – | – | – | – |
| aisle | behavioural | 0.2 | full | 0.68 [0.66, 0.70] | 0.53 [0.46, 0.57] | 0.40 | 0.211 | 0.00 | 0.00 | 3.85 |
| aisle | behavioural | 0.2 | fixed | 0.58 [0.55, 0.60] | 0.26 [0.18, 0.34] | 0.40 | 0.000 | 0.00 | 0.00 | 0.00 |
| aisle | behavioural | 0.2 | oracle | 0.97 [0.93, 1.00] | 0.96 [0.89, 1.00] | 0.94 | 0.470 | 0.00 | 0.90 | 0.00 |
| aisle | behavioural | 0.2 | mlci | 0.69 [0.68, 0.70] | 0.57 [0.56, 0.57] | 0.40 | – | – | – | – |
| aisle | behavioural | 0.3 | full | 0.67 [0.64, 0.70] | 0.50 [0.42, 0.57] | 0.40 | 0.201 | 0.00 | 0.00 | 4.10 |
| aisle | behavioural | 0.3 | fixed | 0.58 [0.55, 0.60] | 0.26 [0.18, 0.34] | 0.40 | 0.000 | 0.00 | 0.00 | 0.00 |
| aisle | behavioural | 0.3 | oracle | 0.97 [0.93, 1.00] | 0.96 [0.89, 1.00] | 0.94 | 0.476 | 0.00 | 0.90 | 0.00 |
| aisle | behavioural | 0.3 | mlci | 0.69 [0.68, 0.70] | 0.57 [0.56, 0.57] | 0.40 | – | – | – | – |
| aisle | violation | 0.0 | full | 0.62 [0.58, 0.65] | 0.36 [0.25, 0.45] | 0.40 | 0.092 | 0.00 | 0.00 | 1.95 |
| aisle | violation | 0.0 | fixed | 0.58 [0.55, 0.60] | 0.26 [0.18, 0.34] | 0.40 | 0.000 | 0.00 | 0.00 | 0.00 |
| aisle | violation | 0.0 | oracle | 0.92 [0.84, 0.98] | 0.92 [0.85, 0.98] | 0.90 | 0.422 | 0.00 | 0.80 | 0.00 |
| aisle | violation | 0.0 | mlci | 0.59 [0.58, 0.60] | 0.50 [0.49, 0.50] | 0.40 | – | – | – | – |
| aisle | violation | 0.05 | full | 0.58 [0.53, 0.64] | 0.39 [0.26, 0.51] | 0.48 | 0.075 | 0.00 | 0.05 | 2.30 |
| aisle | violation | 0.05 | fixed | 0.57 [0.55, 0.60] | 0.24 [0.16, 0.32] | 0.40 | 0.000 | 0.00 | 0.00 | 0.00 |
| aisle | violation | 0.05 | oracle | 0.65 [0.56, 0.73] | 0.74 [0.68, 0.81] | 0.68 | 0.168 | 0.00 | 0.25 | 0.00 |
| aisle | violation | 0.05 | mlci | 0.47 [0.45, 0.49] | 0.21 [0.16, 0.26] | 0.40 | – | – | – | – |
| aisle | violation | 0.1 | full | 0.57 [0.52, 0.62] | 0.41 [0.28, 0.54] | 0.50 | 0.039 | 0.00 | 0.05 | 2.45 |
| aisle | violation | 0.1 | fixed | 0.57 [0.54, 0.60] | 0.22 [0.13, 0.31] | 0.40 | 0.000 | 0.00 | 0.00 | 0.00 |
| aisle | violation | 0.1 | oracle | 0.56 [0.50, 0.64] | 0.70 [0.66, 0.75] | 0.63 | 0.100 | 0.00 | 0.10 | 0.00 |
| aisle | violation | 0.1 | mlci | 0.43 [0.41, 0.45] | 0.10 [0.04, 0.16] | 0.40 | – | – | – | – |
| aisle | violation | 0.2 | full | 0.56 [0.51, 0.61] | 0.45 [0.31, 0.58] | 0.52 | 0.029 | 0.00 | 0.05 | 2.15 |
| aisle | violation | 0.2 | fixed | 0.56 [0.54, 0.59] | 0.21 [0.12, 0.30] | 0.40 | 0.000 | 0.00 | 0.00 | 0.00 |
| aisle | violation | 0.2 | oracle | 0.55 [0.50, 0.60] | 0.69 [0.67, 0.73] | 0.66 | 0.056 | 0.00 | 0.05 | 0.00 |
| aisle | violation | 0.2 | mlci | 0.41 [0.40, 0.42] | 0.01 [0.00, 0.04] | 0.40 | – | – | – | – |
| aisle | violation | 0.3 | full | 0.56 [0.52, 0.62] | 0.40 [0.27, 0.53] | 0.50 | 0.020 | 0.00 | 0.05 | 1.30 |
| aisle | violation | 0.3 | fixed | 0.55 [0.52, 0.57] | 0.15 [0.07, 0.23] | 0.40 | 0.000 | 0.00 | 0.00 | 0.00 |
| aisle | violation | 0.3 | oracle | 0.55 [0.50, 0.61] | 0.68 [0.66, 0.72] | 0.63 | 0.043 | 0.00 | 0.05 | 0.00 |
| aisle | violation | 0.3 | mlci | 0.40 [0.40, 0.40] | 0.00 [0.00, 0.00] | 0.40 | – | – | – | – |

## identifiability

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

## dl_coding_supplement

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
