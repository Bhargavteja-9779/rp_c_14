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

## Stage 3 (revision after external review, written before the corresponding runs)

**Instance-dependent noise model (candidate "v3").** Observation model
P(ỹ = k | z, x_i) = (1 − e_i)[k = z] + e_i · p_i(k) / (1 − p_i(z)) for k ≠ z, with e_i = σ(a + b·u_i),
u_i = 1 − max_k p_i(k) the uncertainty of the current leave-one-out predictive distribution p_i. Flips are thus more
likely for uncertain series and go preferentially to classes the classifier finds plausible. Two parameters (a, b),
fitted inside the same EM; added as a fourth candidate to the BIC choice (none / symmetric / instance / full).
Caveat stated in advance: the synthetic boundary noise of the benchmark flips low-margin series to the runner-up
class, a mechanism close to this model, so a gain under boundary noise is weak evidence; the real-noise ECG data are
the decisive test.

**Design data and adoption rule (fixed now).** Design set of stage 2 (39 datasets, seeds 0–1), conditions sym 0/20/40,
asym 20/40, boundary 20/30. The candidate replaces the stage-2 method only if, relative to it, (i) mean accuracy at
30 % boundary noise rises by ≥ 0.5 points and (ii) no other condition loses more than 0.2 points. Otherwise the
stage-2 method stays the proposed method and the candidate is reported as a negative result.

**New baseline.** Backward loss correction (Natarajan et al. 2013; Patrini et al. 2017) applied to the squared loss:
ridge fitted to the corrected targets onehot(ỹ)·T̂⁻¹, with T̂ estimated by the anchor-point method from calibrated
LOO probabilities of wide-grid ridge (97th-percentile anchors, as in Patrini et al.).

**Additional outcome measures (all datasets, seeds 0–4).** Recall of the smallest training class, worst-class recall,
expected calibration error (15 equal-width bins, top label) and Brier score of probabilities obtained by fitting one
temperature on the final model's LOO decision values against its own training targets (no clean labels used).

**Stage-3 outcomes (design set).** (a) Instance-dependent candidate: +0.69 points at 30 % boundary noise but −0.59
and −0.78 points at 20 % and 40 % asymmetric noise relative to the stage-2 method (`supplementary/design_v4_results.csv`).
Criterion (ii) fails, so it is **not adopted**; it is run on all datasets and on the ECG data and reported as a
negative result. (b) Backward correction: Patrini et al. describe two anchor rules (97th percentile and arg-max).
On 13 design datasets the 97th-percentile rule lost 6.7 points on clean labels relative to ridge and the arg-max rule
0.2 points; the arg-max rule is used, i.e. the baseline is given the better of its two published configurations.
