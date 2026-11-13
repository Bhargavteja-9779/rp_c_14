# Review round 1 — general referee (hostile but fair)

Manuscript version reviewed: first complete build (v1 method as "proposed", wide grid, 108 datasets).

| # | Severity | Issue | Action | Status |
|---|---|---|---|---|
| 1 | **FATAL (for the original framing)** | The proposed method (LOO-EM v1) is *worse* than the properly regularised baseline on clean labels (−1.8 points over 108 datasets, 12 wins / 59 losses) and at 10 % noise. A paper that sells it as an improvement would be wrong. | Research changed, not wording: diagnosed (full C×C matrix behind a one-parameter gate), redesigned as BIC selection among none/symmetric/full on a separate design set, confirmed on 60 untouched datasets. v1 is kept in every table as a negative/trade-off result. Paper reframed as an empirical study whose main result is the regularisation finding. | Fixed |
| 2 | MAJOR | Even after the fix the gain over wide-grid ridge is small (≤ 2.2 points) and there is a significant 0.3-point loss on clean labels. | Stated in abstract, Sect. 6.3 and Discussion; no claim of general superiority. | Disclosed (cannot be "fixed") |
| 3 | MAJOR | "Balanced accuracy and macro-F1 lead to the same conclusions" was asserted without being checked. | Checked. **False for the grid under symmetric noise** (balanced accuracy ±0, macro-F1 −3.1 points at 40 %). New paragraph "The gain depends on the metric", imbalance stratification added, abstract/recommendations/conclusion rewritten. | Fixed (claim replaced by the measured result) |
| 4 | MAJOR | Only one feature transform in the main experiment. | Hydra run added (seeds 0–2); MultiRocket was planned but not completed for memory/time reasons and is **not claimed**. | Partly fixed |
| 5 | MODERATE | No indication of seed-to-seed variability. | Table S (seed standard deviation of the across-dataset mean) added; per-dataset seed means are the unit for all tests. | Fixed |
| 6 | MODERATE | Conditions are undefined when a class disappears after noise injection; the number of datasets therefore varies by column. | n reported per condition in every statistics table; averaging rule stated in the caption of Table 1. | Fixed |
| 7 | MODERATE | Timeline figure of runtime mixes threads. | Main experiment timed with one BLAS thread per worker; stated. Timings on the large datasets used two threads; stated. | Fixed |
| 8 | MINOR | PDF produced without equations (LibreOffice Math component missing). | Component installed, noted in README; DOCX was never affected. | Fixed |
| 9 | MINOR | Statistics table printed p = 0.0000. | Printed as < 0.0001. | Fixed |
| 10 | MINOR | Table 1 unreadable (12 columns with long headers). | Short column codes with a legend in the caption. | Fixed |

Recommendation at this round: major revision.
