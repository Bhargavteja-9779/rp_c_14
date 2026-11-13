# Review round 4 — machine-learning methodology and code/paper consistency

The code was read against the manuscript line by line.

| # | Severity | Paper says | Code does | Action |
|---|---|---|---|---|
| 1 | MODERATE | "ridge classifier as in scikit-learn" | `GramRidge` (dual, eigendecomposition); unit test asserts identical selected penalty and identical predictions to `RidgeClassifierCV`, and LOO values equal to brute-force refits | Consistent; test cited in Sect. 5 |
| 2 | MODERATE | EM stops at max change < 1e-3 or 20 iterations | `loo_em(max_iter=20, tol=1e-3)`; most runs hit the iteration cap (mean ≈ 14–16 per model) | Consistent. Sensitivity (5 and 50 iterations) added so that the cap is shown not to matter |
| 3 | MODERATE | "alternating a one-dimensional search over β and the closed-form update of T" | 5 inner alternations per EM iteration, bounded scalar search on log β ∈ [−4, 6] | Detail added to the supplement protocol |
| 4 | MODERATE | Penalty "re-selected by LOO squared error on the current targets" | v1: exhaustive over the grid every iteration; BIC variant: exhaustive at iteration 0, then local search around the previous optimum, window extended while the optimum is on its edge | Described in Sect. 4.3; equivalence verified on a test case (same penalty, same likelihood) |
| 5 | MODERATE | BIC compares ℓ − (k/2) log n with k = 0, 1, C(C−1) | `loo_em_bic`: exactly this; the noise-free likelihood uses β fitted on first-round LOO scores | Consistent |
| 6 | MODERATE | Hydra features | aeon `HydraTransformer`, then square-root and column standardisation — **not** the sparse scaler of the original Hydra pipeline | Deviation now stated in Sect. 6.6 |
| 7 | MINOR | "ENN with k = 3" | Sample kept if its label wins the vote or gets ≥ 50 % of votes | Tie rule noted in supplement |
| 8 | MINOR | "flagged series removed" (confident learning) | `cleanlab.filter.find_label_issues(filter_by="prune_by_noise_rate")`; fall-back to arg-max disagreement if a class is absent; a class is never emptied | Noted in supplement |
| 9 | MINOR | Eq. (2) intercept term | implemented as + 1/n in `hat_diag` and column mean in `loo` | Consistent |
| 10 | MINOR | Very short series | padded by edge replication to the minimum length the transform accepts | Noted in supplement |
| 11 | MODERATE | Boundary noise "smallest clean LOO margin" | computed from wide-grid ridge on the clean labels of the same resample; ties broken by tiny Gaussian jitter from the noise RNG | Consistent |

Optimisation/training protocol of the deep baselines (100 epochs, Adam 1e-3, no early stopping, Co-teaching warm-up of 10 epochs with the true noise rate) matches Sect. 5. No contradiction between code and manuscript remains to my knowledge.
