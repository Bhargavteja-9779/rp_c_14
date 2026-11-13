# Final consistency audit, reproducibility audit, rejection-risk report and self-decision

## 1 Consistency (title ↔ abstract ↔ contributions ↔ method ↔ code ↔ experiments ↔ tables ↔ figures ↔ statistics ↔ conclusion)

| Check | Result |
|---|---|
| Every number in the text is generated from `results/` at build time | Yes (`<<…>>` expressions; build aborts on an unresolved one). Build output: "consistency checks passed" |
| Independent recomputation of headline numbers from `raw_results.csv` | Matched (+4.24; +2.20 / +1.26 / −0.31; 98 %) |
| Every table and figure cited in the text; numbering automatic | Yes (10 tables, 10 figures; checked by the builder) |
| Every citation has a reference entry and vice versa | Yes (42 / 42) |
| Equations ↔ implementation | Eq. 1–3: `GramRidge` (unit tests against brute-force LOO and scikit-learn). Eq. 4 / Prop. 1: unit test. Prop. 2 counter-example recomputed. Eq. 5 and E/M steps: `cleaning.loo_em`; BIC: `cleaning.loo_em_bic` |
| Every experiment mentioned exists in `results/raw/` | main (108), BIC pass (108), calibrated-CL pass (108), large (3), Hydra (29), UEA (18), families (108 × 2 methods), deep (6), sensitivity (30 × 4 kernel counts). Not run and not claimed: MultiRocket, ElectricDevices, PhonemeSpectra |
| Failed runs | 0 of 144,785 main rows carry an error message |
| Abstract length | 245 words |
| Title matches content | Yes: the main result is about regularisation versus relabelling |

## 2 Reproducibility (as a stranger)

| Question | Answer |
|---|---|
| Install? | `pip install -r requirements.txt` (pinned versions) or `environment.yml` |
| Obtain data? | Automatic per-dataset download through aeon; no manual step |
| Run the pipeline? | `python run_all.py --mode quick` (≈ 2 min, verified) / `--mode full` (resumable) / `--mode report` |
| Regenerate tables, figures, statistics? | `--mode report` (≈ 40 s on existing raw results; verified repeatedly) |
| Regenerate the exact manuscript numbers? | `python manuscript/build_manuscript.py` from the committed `results/` — yes. From scratch: accuracies are deterministic given library versions; timings are not |
| Known obstacles | Full run needs ≈ 15–20 CPU-hours and ≈ 20 GB of disk for the feature cache; Hydra needs > 8 GB for long series; PDF needs `libreoffice-math`. `--mode full` as written also attempts MultiRocket and uses the full dataset list for Hydra, which exceeded this machine's memory — the committed results used the 30-dataset subset for Hydra; the README states this |
| Not reproducible | The interim-look history itself (documented in `supplementary/protocol.md`, with design-stage CSVs) |

## 3 Rejection-risk report

| # | Risk | Evidence | Severity | Mitigation | Fixed? |
|---|---|---|---|---|---|
| 1 | Novelty | EM with a transition matrix and BIC are classical; LOO identity is shared with PreVal; "widen the grid" is simple | **High** | Honest framing as an empirical study; ablations show why each ingredient is needed; two earlier topics were abandoned for prior art | No — inherent |
| 2 | Methodology | Proposed method redesigned after an interim look | Medium | Confirmation set of 60 untouched datasets; full disclosure | Mitigated |
| 3 | Dataset | Synthetic noise; UCR/UEA only; archive duplicates; possible subject leakage | **High** | Three noise models incl. an adversarial one; duplicate audit and re-analysis; limitations section | Partly |
| 4 | Statistical | Small effects; several confirmation-set comparisons not significant; Holm per pair only | Medium | Effect sizes, CIs, W/T/L everywhere; conclusions limited to pre-specified pairs | Disclosed |
| 5 | Reproducibility | Long full run; memory limits; restarts during the study | Low | Resumable per-dataset outputs; raw results committed; quick mode | Mostly |
| 6 | Baselines | No published deep noisy-label TSC method rerun; simple deep baselines on six datasets, where oracle Co-teaching wins at 30 % symmetric noise | **High** | Stated; calibrated confident learning added after review | Partly |
| 7 | Generalisation | Hydra on 29 datasets; no MultiRocket; three large datasets; multivariate under-powered | Medium | Reported as run; no claims beyond | Partly |
| 8 | Writing | Long (≈ 11,400 words incl. tables); dense tables | Low–medium | Supplement carries 31 tables; could shorten on request | Partly |
| 9 | Formatting | Live author guidelines not fetched; reference style (author–year vs numbered), abstract/keyword limits, declaration wording unverified | Medium | Generic Springer Nature format followed; **must be checked by the author on the submission page** | No |
| 10 | Citation | Seven entries with caveats (page ranges, preprints) | Low | Listed in `citation_audit.md` | Partly |
| 11 | Journal scope | Journal mostly publishes deep-learning method papers | Medium | Scope formally fits; cover letter positions the paper | Disclosed |
| 12 | Journal-specific reviewer | Likely requests: real noisy data, GPU baselines, larger gains | **High** | Cannot be met with the available resources | No |
| 13 | Integrity / compliance | Author details, funding, contributions and the AI-use disclosure are placeholders. The study design, code, experiments and text were produced with an AI system; Springer Nature policy requires disclosure and does not accept AI systems as authors. The human authors must verify the work and take responsibility for it | **High if ignored** | Placeholder in Declarations and cover letter | Requires author action |
| 14 | Acceptance timeline | Journal median 68 days in 2026, but 44 % of accepted papers took longer than 72 days; a major-revision outcome is likely here | Medium | None | No |
| 15 | JCR status | Quartile/JIF taken from publisher snippet and aggregators, not from Clarivate | Medium | Author to confirm in JCR and the Master Journal List | No |

No claim of zero risk or of acceptance is made.

## 4 Self-decision

**READY FOR SUBMISSION — conditionally, as an empirical study, after the author-side items below.**

The experiments that the manuscript reports are complete, internally consistent and reproducible from the supplied code, and no known error remains. Scientifically, the honest expectation is "sent to review, major revision or rejection on novelty/baselines" rather than quick acceptance; the three high-severity risks (novelty, synthetic noise, missing deep baselines) cannot be reduced further on a CPU-only machine without real noisy-label data.

Before submitting, a human author must:

1. confirm the journal's Q1 status and impact factor in Clarivate JCR and check the Master Journal List;
2. open the live submission guidelines and adjust reference style, abstract/keyword limits and declarations if they differ;
3. fill in authors, affiliations, funding, contributions and competing interests, and write the AI-use disclosure truthfully;
4. read the manuscript and code critically and take responsibility for every claim — in particular Propositions 1–2 and the protocol history;
5. deposit the code and raw results in a public repository and insert the URL;
6. ideally, add at least one dataset with verified real label errors and one published deep noisy-label baseline run on a GPU, which would address the two requests reviewers are most likely to make.
