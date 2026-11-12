# Review round 8 — a competing researcher

Position: "I work on noisy-label learning; this paper undersells my methods and oversells a ridge trick."

| # | Severity | Attack | Is it valid? | Response |
|---|---|---|---|---|
| 1 | **MAJOR** | Confident learning was run with uncalibrated probabilities; a weakened baseline. | **Yes.** | New run with a fitted temperature (108 datasets × 5 seeds). It improves the baseline by 1–2.5 points and makes it *better than wide-grid ridge at 30–40 % symmetric noise*, on a par with LOO-EM there. The manuscript now uses the calibrated baseline everywhere, reports the uncalibrated one as a cautionary result, and no longer says that cleaning "never" helps. LOO-EM's claimed advantage over confident learning is reduced to (i) clean and lightly noisy data and (ii) cost. |
| 2 | MAJOR | The improvement attributed to LOO-EM is really the grid. | Checked by design. | Every cleaning method, including LOO-EM, uses the wide grid; LOO-EM is always compared with wide-grid ridge. The ablation `bic_std_grid` isolates the grid inside LOO-EM. |
| 3 | MAJOR | The benchmark favours a linear model: tiny training sets on which deep noisy-label methods cannot work. | Partly. | Stated in Limitations; three large datasets added; no claim is made against deep noisy-label methods. |
| 4 | MAJOR | The noise models favour a class-conditional EM. | Yes for sym/asym; no for boundary. | Boundary noise results (no gain) are reported in abstract-level text, results, and limitations. |
| 5 | MODERATE | Co-teaching was given the true noise rate but only a 10-epoch warm-up and no tuning. | Fair. | Described as a simple baseline on six small datasets "supporting no general claim". |
| 6 | MODERATE | Hyper-parameters of LOO-EM were tuned; baselines were not. | Partly. | LOO-EM's settings were fixed on development/design data and their sensitivity is reported; ENN (k = 3), confident learning (cleanlab defaults, 5 folds) use standard defaults. Stated. |
| 7 | MODERATE | "Wide grid" could be achieved by any practitioner; is it research? | It is a finding, not an invention. | Presented as such, with a proof of why, the size of the effect, where it fails (n < 100, ≥ 10 classes at heavy asymmetric noise) and its metric caveat. |
| 8 | MINOR | PreVal-style logistic head instead of ridge might be more robust. | Unknown. | Listed as future work; not tested. |
