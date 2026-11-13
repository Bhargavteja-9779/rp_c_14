# Review round 5 — dataset validity, generalisation, attempts to invalidate the central claims

| # | Severity | Attack | Evidence gathered | Outcome |
|---|---|---|---|---|
| 1 | **MAJOR** | "The wide grid only helps because UCR training sets are tiny." | Stratification by n and a separate run on the three largest feasible UCR datasets (3,601–7,200 training series). | The effect is *larger* on large sets (e.g. +15 to +18 points at 40 % noise) and absent or negative below 100 series. Claim refined, not invalidated. |
| 2 | **MAJOR** | "The accuracy gain is a majority-class artefact." | Balanced accuracy, macro-F1, stratification by imbalance ratio. | **Partly valid.** Under symmetric noise the gain vanishes on class-balanced metrics and is negative for macro-F1 at 40 %. Central claim rewritten accordingly. |
| 3 | MAJOR | "Results are specific to MiniRocket." | Hydra features, seeds 0–2. | Reported in Sect. 6.6 as measured. MultiRocket not run. |
| 4 | MAJOR | "Univariate only." | 18 multivariate UEA datasets. | Same direction, low power; reported descriptively. |
| 5 | MAJOR | "Synthetic class-conditional noise flatters an EM that assumes class-conditional noise." | Boundary (instance-dependent) noise designed to be adversarial to margin-based methods. | LOO-EM gains nothing there; stated plainly in abstract-level summary, Sect. 6.5 and Limitations. No real-noise benchmark exists to our knowledge; this remains the main external-validity gap. |
| 6 | MODERATE | "Train/test contamination in the archive inflates everything." | Duplicate audit; re-analysis without the affected datasets. | Conclusions unchanged in direction. |
| 7 | MODERATE | "Subject/recording leakage." | Not auditable from the archive files (no subject identifiers). | Listed as a limitation of the benchmark, affects all methods equally. |
| 8 | MODERATE | "Deployment: can a practitioner know which regime they are in?" | Estimated noise rate is conservative (≈ 0.15 at a true 0.20); BIC accepts a noise model in ≈ 28 % of clean runs. | Reported; recommendations in the Discussion are conditional on prior knowledge and say so. |
| 9 | MINOR | PhonemeSpectra excluded from the UEA run. | It has 3,315 training series, above the stated 3,000 limit. | Exclusion rule stated; not outcome-dependent. |
