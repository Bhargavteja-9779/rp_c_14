# Verification log

## How the acceptance timeline was verified
`scripts/journal/jk.py` pages through the Crossref REST API for ISSN 2213-1248 (articles published since
2025-01-01), reads the `received` and `accepted` assertions deposited by Springer Nature and reports the
distribution of (accepted − received) by acceptance half-year:

| Accepted in | n | Median | Q1 | Q3 | P90 | ≤ 72 d | ≤ 60 d |
|---|---|---|---|---|---|---|---|
| 2025 H1 | 138 | 78.5 | 63 | 109 | 126 | 42 % | 20 % |
| 2025 H2 | 306 | 90 | 71 | 118 | 143 | 28 % | 15 % |
| 2026 H1 | 549 | 68 | 52 | 95 | 126 | 56 % | 38 % |
| 2026 H2 (to 10 Nov 2026) | 355 | 68 | 53 | 91 | 112 | 56 % | 37 % |

1,348 of 1,354 articles carried both dates. This is *submission-to-acceptance of accepted papers*; it is not
first-decision time, not acceptance-to-publication time, and says nothing about rejection probability.

## What was NOT verified (be explicit)
* JCR quartile, category and the exact JIF: taken from the publisher page as rendered in a search snippet and
  from aggregator sites; Clarivate's JCR and Master Journal List require a login and were not accessed.
* Acceptance rate: no reliable figure found; none is quoted.
* The live Springer submission-guideline page and APC page: bot-protected; details came from search snippets.
* No report of a Web of Science "on hold"/delisting action was found in a targeted search, but absence of
  search results is not proof; the Master Journal List should be checked on the day of submission.

## Legitimacy
Publisher: Springer Nature. Owner: King Saud University. DOIs resolve through Crossref with full
received/accepted metadata. Publication volume is high (about 900 acceptances in 2026 to date), which editors of
slower journals sometimes view sceptically; this is disclosed here rather than hidden.

## Scope evidence
A random sample (seed 0) of 60 titles accepted in 2026 includes time-series classification and anomaly detection
("Interpretable time-frequency collaboration … classification of long time series", "TFD-CAD … time series
anomaly detection"), robust/federated learning, explainable classical ML ("ASP-KNN …") and data-quality topics,
i.e. work comparable in kind to this manuscript. Most accepted papers are deep-learning methods evaluated on
GPUs; a CPU-scale statistical-learning paper is less typical and is listed as a reviewer risk.
