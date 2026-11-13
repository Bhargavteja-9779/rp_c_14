# Experimental protocol (written before the full run)

**Development / held-out separation.** All design decisions of the proposed method (wide penalty grid, soft
targets, full transition matrix with symmetric shrinkage, likelihood gate with factor 1.0) were taken on 12
development datasets drawn with `numpy.random.RandomState(2026).choice(sorted(112 UCR datasets), 12,
replace=False)`, default split only, noise seeds 100–102. The remaining 100 datasets were not looked at before
the configuration in `config/default.yaml` was frozen. Results are reported for all 112 and separately for the
100 held-out datasets.

**Development history (kept, including negative results).**
1. Hard relabelling of every LOO disagreement (iterated) *reduced* mean dev accuracy on clean data from 0.858
   to 0.802 and at 20 % symmetric noise from 0.773 to 0.744 → rejected as main method, kept as ablation.
2. Observation: the default penalty grid `logspace(-3,3,10)` saturates at its upper end under noise (mean
   selected log10(alpha) 2.8–3.0) → hypothesis "the fragility is under-regularisation" → grid widened to
   `logspace(-3,9,25)`; this alone recovered +2 to +5 points on dev datasets with n ≥ 100.
3. EM with a class-conditional noise model on LOO predictive probabilities; gains concentrated on asymmetric
   noise. A likelihood gate was added after EM without gate lowered accuracy on tiny clean datasets.

**Resampling.** Seeds 0–4, fixed in advance; seed 0 is the archive split, seeds 1–4 are stratified resamples
with the original train/test sizes. Noise realisation is a deterministic function of (dataset, seed, noise
model, rate). Noise is applied to training labels only; test labels are always clean.

**Noise models.** symmetric (0, 10, 20, 30, 40 %), asymmetric pair-flip c → c+1 mod C (10–40 %), and
boundary/instance-dependent (10–30 %): the lowest clean-LOO-margin samples are relabelled to their runner-up class.
The boundary model is adversarial to margin-based detectors by construction.

**Primary outcome.** Test accuracy on clean labels, averaged over seeds per dataset. Primary comparison:
`loo_em` vs `ridge_std` and vs `ridge_wide`, per noise condition, two-sided Wilcoxon signed-rank over datasets,
Holm-corrected across the 12 conditions. All other comparisons are secondary.

**Leakage audit.** (i) feature transforms are fitted on training series only (MiniRocket bias quantiles);
(ii) scaling statistics from training features only; (iii) penalty and all EM quantities use training labels
only; (iv) resamples are stratified re-partitions, so no series is in both sets; (v) known archive caveat: some
UCR datasets contain series cut from the same recording/subject in train and test — this affects all methods
equally and absolute accuracies should be read with that in mind; (vi) exact duplicate series across the
train/test boundary are counted by `scripts/leakage_audit.py`.

## Stage 2 (added during the full run, before any confirmation-set result was inspected)

**What happened.** When 36 of the 108 non-large datasets had finished (alphabetical order, ACSF1 … FaceAll), an
interim look showed that the frozen method (v1: full C x C transition matrix, acceptance gate with a
one-parameter BIC penalty) *lowered* accuracy on clean labels by 2.2 points (Wilcoxon p = 0.0002, 4 wins / 18
losses) and at 10 % noise, and helped only at >= 30 % asymmetric noise. The development set contained no
dataset with more than 8 classes, so it had not exposed the problem.

**Diagnosis.** The gate charged one parameter for a model with C(C-1) free parameters, so on many-class data the
transition matrix absorbed ordinary model error as "label noise".

**Revision (v2, `loo_em_bic`).** Fit both the symmetric (1 parameter) and the full (C(C-1) parameters) noise
model and select among {no noise, symmetric, full} by BIC on the leave-one-out observed-label log-likelihood.
No other change. Design set for this decision: the 39 datasets listed in `supplementary/design_set_v2.txt`
(seeds 0–1; results in `supplementary/design_v2_results.csv`). On it v2 changed mean accuracy relative to
wide-grid ridge by −0.2 (clean), +1.3 (sym 20 %), +1.7 (sym 40 %), +1.5 (asym 20 %), +0.6 (asym 40 %) points.

**Rejected alternative (negative result).** Anchor-point initialisation of T (Patrini et al., 2017) from the 3 %
most confident calibrated LOO predictions was tried to recover v1's gain under heavy asymmetric noise; on the
design set it was worse than wide-grid ridge on clean labels (−1.5 to −2.5 points) and under symmetric noise
(`supplementary/design_v3_results.csv`). It is not part of the final method.

**Confirmation set.** The 112 − 39 − (development datasets not already in the design set) datasets that were in
neither the development set nor the design set. v2 was frozen before any of their results was inspected. The
manuscript reports v1 in full as a negative/trade-off result and reports v2 separately on the confirmation set.
Primary comparison (updated): `loo_em_bic` vs `ridge_wide` and vs `ridge_std` on the confirmation set.
