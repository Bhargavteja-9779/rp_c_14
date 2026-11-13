# Review round 9 — Associate Editor, Journal of King Saud University – Computer and Information Sciences

**Scope.** Machine-learning methodology for time series classification and data quality: in scope. Most recent articles are deep architectures with GPU experiments; this is a statistical-learning/benchmark paper, which is less typical for the journal but not outside it.

**Would it survive peer review here?** Possibly, not certainly. Strengths: 108 + 18 + 3 datasets, five resamples, pre-specified seeds, held-out confirmation set, effect sizes with intervals, negative results reported, full code. Weaknesses a referee will raise:

| # | Severity | Concern | Status |
|---|---|---|---|
| 1 | MAJOR | The headline contribution is an empirical finding plus a modest method; referees who expect a new architecture with large gains may call it incremental. | Cannot be engineered away without overclaiming. Cover letter positions the paper as an empirical study with a practical recommendation. |
| 2 | MAJOR | No comparison with published deep noisy-label TSC methods. | Disclosed; a referee may still request it (would need GPU resources). |
| 3 | MAJOR | Synthetic noise only. | Disclosed. |
| 4 | MODERATE | Length: about 11,000 words including tables, 10 figures, 9 tables. Springer imposes no hard limit for this journal that we could verify; some figures could move to the supplement on request. | Noted. |
| 5 | MODERATE | Author guidelines could not be fetched directly (bot protection). | Manuscript follows the generic Springer Nature journal guideline; item-by-item check required at submission (final risk report). |
| 6 | MODERATE | Declarations incomplete (placeholders). | Authors must complete funding, contributions, competing interests, AI-use disclosure. |
| 7 | MINOR | Title is a slogan. | Kept; a descriptive subtitle follows the colon. |
| 8 | MINOR | British spelling throughout; consistent. | OK. |

Editor's summary: technically careful and honest; novelty moderate; decision after review would most plausibly be "major revision" with requests for real-noise data and deep baselines.
