# Pre-specified protocol: real-world annotation noise (PhysioNet/CinC Challenge 2017)

Written before any method was run on these data.

**Data.** The 8,528 public training recordings of the 2017 PhysioNet/Computing in Cardiology Challenge
(single-lead AliveCor ECG, 300 Hz, 9–61 s; classes Normal N, AF A, Other rhythm O, Too noisy ~).

**Noisy labels (training).** `REFERENCE-v0.csv`: the labels released with the data. According to the challenge
paper (Clifford et al. 2017) the initial labels were produced by an outsourced annotation company with about 10 %
over-read by experts.

**Reference labels (evaluation).** `REFERENCE-v3.csv`: the final labels after (i) a visual re-check of all
recordings for unreadable signals (V2) and (ii) expert relabelling (V3). 412 training labels (4.8 %) differ from
v0. V3 is *expert-revised*, not error-free: the paper reports low inter-expert agreement (Fleiss κ ≈ 0.25) on the
contentious recordings, and most training recordings were never re-read individually. Results are therefore
"agreement with the expert-revised reference", and the detection analysis measures recovery of *documented*
corrections, a lower bound on true error recovery.

**Preprocessing.** Each record is z-normalised, decimated to 150 Hz (scipy `decimate`, q = 2, zero-phase IIR) and
brought to 30 s (4,500 samples): longer records are centre-cropped, shorter ones are tiled. No other filtering.

**Splits.** Stratified (on the v0 labels, the only labels a practitioner would have) 5-fold cross-validation,
repeated twice (fold seeds 0 and 1). Every transform, scaling statistic, penalty and EM quantity is fitted on the
training folds only.

**Methods.** Exactly the frozen configurations of the UCR study (`config/default.yaml`), plus the methods added in
the revision (backward-corrected ridge, the margin-dependent noise model if it passes its design-set stage, and the
CTW re-implementation if computationally feasible), each trained on v0 labels. Two references are also reported:
ridge (wide grid) trained on v3 labels (attainable with expert labels) and the same evaluated against v0 test labels.

**Outcomes.** Primary: accuracy and the challenge's F1 score (mean of the per-class F1 for N, A and O) against v3
test labels, from pooled out-of-fold predictions. Secondary: balanced accuracy, macro-F1 over all four classes,
per-class recall; label-error detection AUROC of each method's suspicion score against the indicator v0 ≠ v3 on
training folds; fraction of training labels agreeing with v3 after cleaning.

**Primary comparisons.** LOO-EM (BIC) vs ridge (wide grid); ridge (wide grid) vs ridge (default grid).
**Tests.** McNemar's exact test on pooled out-of-fold correctness (each record contributes once per repetition;
reported per repetition), and a paired bootstrap over records (10,000 resamples) for the accuracy and F1
differences. Holm correction over the two primary comparisons. No other comparison is used for a claim.
