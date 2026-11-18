# Response to the external review (pre-submission)

We thank the reviewer for a careful and constructive report. Every point is answered below, with the change made and
where to find it. Numbers are generated from the result files (`python scripts/build_response.py`).

The most consequential change is the one the reviewer asked for first, a test on real annotation errors. It did not
confirm the benefit of label correction that the synthetic experiments suggested, and the paper has been rewritten
around that finding rather than around the proposed method. We explain below why we think this makes the paper
stronger, not weaker.

---

## Issue 1 — No real-world noisy labels (highest priority)

**Done.** We searched for public time-series data with *paired* original and independently verified labels. The two
multi-scorer sleep datasets we identified (Dreem DOD-H/DOD-O; ISRUC-Sleep) could not be obtained: the Dreem storage
buckets no longer exist and the ISRUC server did not respond. We found and used the PhysioNet/CinC 2017 challenge data:
8,528 single-lead ECG recordings whose released labels were produced by an outsourced annotation company
(about 10% over-read) and later revised by the organisers and experts; 412 labels (4.8%) differ.
This is the situation the reviewer describes for industrial data — operational labels, later partially verified —
in a domain where the data are public.

Protocol written before any run (`supplementary/protocol_cinc2017.md`): train on original labels, evaluate against
expert-revised labels, 5-fold CV × 2 repetitions, McNemar's exact test and paired bootstrap, Holm over two primary
comparisons. Results (Sect. 6.6, Tables and Fig. in that section):

* Training on the noisy rather than the expert labels costs 0.6 accuracy points.
* LOO-EM **lowers** agreement with the expert labels: −1.5 accuracy points, 95% CI −2.1 to −0.9; F1 −0.028 (primary comparison,
  Holm-adjusted p < 0.001). It changed 15.7% of training labels
  where 4.8% were documented as wrong — the identifiability failure anticipated in Sect. 4.4.
* No relabelling or removal method helped; backward loss correction was the only correction that did
  (+0.4 accuracy points, 95% CI +0.1 to +0.6; F1 +0.005).
* Detection of the documented errors: AUROC 0.64–0.70 for all scores.

**On the suggested industrial study.** We agree it would be the ideal next test, but we have no access to technician
labels with maintenance-verified ground truth, and public bearing datasets (CWRU, Paderborn) carry laboratory labels
without natural noise, so they would add synthetic noise again. We therefore did not adopt an "industrial" title or
claim; Limitations states this explicitly.

## Issue 2 — LOO-EM's improvement is modest

**Agreed, and the real data sharpen it.** As suggested, the paper no longer presents LOO-EM as its main contribution:
the abstract, introduction (contributions 1–5) and conclusion now lead with the theory, the regularisation/criterion
finding and the real-noise evaluation. Win/tie/loss counts are reported for every comparison (statistics tables).
When to activate LOO-EM: Sect. 4.5 and recommendation (4) in Sect. 7 — only for accurate classifiers with plausible
heavy class-conditional noise, with the BIC choice and the number of changed labels reported to the user as a
warning signal.

## Issue 3 — Insufficient deep / time-series-specific comparison

**Done within CPU limits.** (a) CTW (IJCAI 2023), a time-series-specific noisy-label method, re-implemented from the
official code with the authors' hyper-parameters (300 epochs) and run on exactly our splits and noise on the UCR
datasets of the CTW paper's own list (seven of eight; MelbournePedestrian has unequal-length series) — Sect. 6.8.
Running the official code directly was not possible in our environment; the re-implementation follows it line by line
and the deviations are listed in `src/models/ctw.py`. (b) Backward loss correction (Patrini et al. 2017) on all 108
datasets and on the ECG data. (c) FCN with cross-entropy, GCE and Co-teaching (supplement). Claims relative to methods
we did not run (SREA, Scale-teaching, newer methods) are explicitly not made.

## Issue 4 — Scalability

**Done.** A primal route (decomposing the p × p scatter matrix when n > p) was implemented and verified to give
identical LOO values and predictions (unit test). A scaling experiment on ElectricDevices (500–14,000 training series,
two feature dimensions, peak memory measured in isolated processes) is reported in Sect. 6.9 with a new figure; e.g.
with about 2,000 features, n = 14,000 trains in 24 s with 1.4 GB peak memory.
Streaming: retraining after label corrections needs neither a new transform nor a new decomposition (Sect. 6.9);
incremental updates for new series are discussed as future work.

## Issue 5 — Instance-dependent noise

**Done, negative result.** An instance-dependent noise model (flip probability rising with the LOO uncertainty, flips
towards plausible classes) was added as a fourth BIC candidate under an adoption rule fixed in advance. It gained
0.7 points at 30% boundary noise on the design set but lost up to 0.8 points under asymmetric noise, failed the rule,
and is reported as rejected (Sect. 6.4). On the ECG data it behaved like LOO-EM.

## Issue 6 — Practical deployment story

**Done.** New end-to-end deployment figure with measured numbers (Sect. 6.9): training time, peak memory, model size
(253 kB) and single-recording inference latency (10 ms on one CPU thread),
and a workflow in which the ranked list of suspicious labels feeds expert review rather than automatic relabelling —
the use the real-data results support.

## Section-by-section suggestions

| Suggestion | Change |
|---|---|
| Title | Now states the contribution and the real-noise evaluation; we did not add "industrial" (no industrial data) |
| Abstract | Restructured: problem → gap → theory/method → synthetic results → real-noise results → implication; ≤ 250 words |
| Introduction | Motivated by a documented case of operational annotation error; RQ5 added; contributions rewritten |
| Related work | Expanded with verified 2024–2026 work (temporal label noise, EHR, soft-label refurbishment, Con4m, medical review, instance-dependent noise) |
| Mathematical formulation | Corollary 1 (label smoothing is a no-op for ridge), primal route, notation checked |
| Methodology | New Sect. 4.5 "From the model to practice"; clearer Fig. 1 separating E-step, classifier M-step, noise-model M-step and BIC |
| Experimental setup | Real data, new baselines, class-wise and calibration metrics, McNemar/bootstrap for paired recordings |
| Results | Class-wise recall and calibration (Sect. 6.5), ECG (6.6), penalty criterion (6.7), CTW (6.8), scaling and deployment (6.9) |
| Discussion | "What the real data changed"; recommendations rewritten from the real-data evidence |
| Limitations | One real-noise dataset; no industrial data; CTW re-implemented; weak base classifier on ECG |

## Critical technical points

* **Standardisation inside LOO.** Exact LOO with per-fold re-standardisation computed for the
  41 datasets with ≤ 100 training series: decision values correlate at 1.000 (median) with the closed form, and
  LOO-EM accuracy does not change significantly (Sect. 6.4).
* **Per-class precision/recall, minority recall, confusion matrices, calibration.** Added (Sect. 6.5, ECG confusion
  matrices in Sect. 6.6). These revealed that the wide grid costs the worst class heavily under symmetric noise
  (worst-class recall 41.0% → 29.8% at 40%), which led to the exploratory penalty-criterion analysis (Sect. 6.7).
* **Placeholders** (authors, affiliations, funding, contributions, competing interests, AI-use statement, repository
  URL) must be completed by the authors; we cannot fill them.
* **References**: every new reference was checked against Crossref or arXiv (`REVIEW/citation_audit.md`).

## Why we did not "make the method win"

The reviewer asked for real-world evidence; we obtained it under a protocol fixed beforehand, and it went against the
proposed method. Re-tuning LOO-EM on the ECG data until it won would have been post-hoc and would not survive scrutiny.
We instead report the result, explain it (Sect. 4.4, 6.6, 7), and turn it into the paper's central practical message:
conclusions drawn from synthetic label noise — which is how most noisy-label methods, including ours, are evaluated —
did not transfer to real annotation errors. We believe this is a more useful and more defensible contribution for the
journal's readership than another method with a small benchmark gain.
