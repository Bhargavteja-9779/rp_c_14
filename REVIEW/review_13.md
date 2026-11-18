# Review round 13 — statistics and protocol of the revision

| # | Severity | Issue | Status |
|---|---|---|---|
| 1 | MAJOR | Several analyses were added after the first results were known. | Each was registered in `supplementary/protocol.md` before it was run (stages 3 and 4), with an adoption rule where a decision was involved; the rejected instance model and the exploratory status of the criterion analysis are reported as such. |
| 2 | MODERATE | Many comparisons are reported; Holm correction is applied within families (12 noise conditions per method pair; 2 primary ECG comparisons). Without a global correction some secondary "significant" results may be false positives. | Stated; claims rest on pre-specified comparisons. |
| 3 | MODERATE | Class-wise metrics are averaged over datasets with very different class counts. | Means and paired tests over datasets, as for accuracy; stratification by number of classes and by imbalance in the supplement. |
| 4 | MODERATE | ECE depends on the binning (15 equal-width bins). | Fixed in advance; Brier score reported alongside as a bin-free measure. |
| 5 | MINOR | Bootstrap for the ECG F1 uses 1,000 resamples (cost). | Stated in Sect. 5. |
| 6 | MINOR | The exact-LOO analysis covers only datasets with ≤ 100 training series (n⁴ cost). | Stated; these are the datasets where the approximation is largest. |

No new statistical error found in the code paths added in the revision (`src/evaluation/realworld.py`, `extra.py`); the
ECG aggregation was checked by recomputing accuracy, F1 and McNemar counts for one fold by hand during development.
