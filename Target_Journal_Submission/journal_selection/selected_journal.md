# Selected target journal

| Field | Value | Source / verification status |
|---|---|---|
| TARGET JOURNAL | Journal of King Saud University – Computer and Information Sciences (JKSUCIS) | — |
| PUBLISHER | Springer Nature (journal no. 44443; owned by King Saud University; published by Elsevier until end of 2024) | Springer journal page (via search-engine snippet) |
| ISSN | 2213-1248 (electronic), 1319-1578 (print, legacy) | Crossref journal record |
| JCR STATUS | Indexed in SCIE / Journal Citation Reports | KSU news item; third-party listings. **Not checked in Clarivate's own JCR/Master Journal List (login/bot-protected) — the author must confirm before submission.** |
| JCR QUARTILE | Q1, Computer Science, Information Systems (reported by aggregators for recent JCR years) | **Aggregator-reported only; not independently verified in JCR.** |
| JOURNAL IMPACT FACTOR | 6.4 (2025 JIF), 5-year 6.0, as shown on the Springer journal page | Search-engine snippet of the publisher page; aggregators give 6.1 for the previous year |
| JCR SUBJECT CATEGORY | Computer Science, Information Systems | Aggregator-reported |
| VERIFIED REVIEW/DECISION TIMELINE | Not verified (publisher page could not be fetched) | — |
| VERIFIED ACCEPTANCE TIMELINE | **Measured by us from Crossref "received"/"accepted" dates: median 68 days for the 904 articles accepted in 2026 (IQR about 52–95 days; 56 % within 72 days, 37 % within 60 days; 90th percentile about 112–126 days).** 2025: median 79 days (H1, n=138) and 90 days (H2, n=306). | `scripts/journal/jk.py` (re-runnable) |
| APC | £1,590 / $2,090 / €1,790 + taxes (Springer page via search snippet); other sites list lower, probably outdated, figures | Must be re-checked at submission; waiver requests must be made at submission |
| SCOPE FIT | High. Recent issues contain time-series classification, anomaly detection, federated/robust learning and data-quality papers | 60 randomly sampled 2026 titles (see journal_verification.md) |
| RESEARCH FIT | High: a machine-learning methodology paper with large-scale benchmark evidence | — |
| KEY AUTHOR GUIDELINES | Springer Nature journal format: editable source (Word/LaTeX), structured title page, abstract, keywords, numbered sections, Declarations (funding, competing interests, data/code availability, author contributions). **The full guideline page could not be fetched (bot protection); the manuscript follows the generic Springer Nature guideline and must be checked against the live page.** | — |
| KEY REJECTION RISKS | (1) perceived incrementality (EM for label noise is classical); (2) no GPU-scale deep baselines; (3) synthetic noise only; (4) 44 % of accepted papers took longer than 72 days — the timeline is a median, not a promise | — |

## What the user's constraints could and could not be satisfied

* **Q1, non-MDPI, reputable publisher** – satisfied subject to the author confirming the quartile in JCR itself.
* **≤ 72 days to final acceptance** – satisfied *at the median* (68 days in 2026). It is **not guaranteed**: roughly 4 in 10 accepted papers in 2026 took longer, and rejected papers are not in the data at all.
* **≈ 30 days** – not satisfied by any reputable Q1 journal we measured; only about 5 % of JKSUCIS papers are accepted within 35 days.
* **Oxford University Press** – every OUP journal we could measure is slower (Briefings in Bioinformatics 120, Bioinformatics 152, JAMIA 115, PNAS Nexus 168, National Science Review 122 days median; J. Comput. Design & Eng. self-reports 15 weeks). *Biology Methods and Protocols* (OUP) measured 57 days but on only 32 papers, is not a computer-science venue and its Q1 status was not established.
