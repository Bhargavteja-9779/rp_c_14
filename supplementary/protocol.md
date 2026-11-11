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
