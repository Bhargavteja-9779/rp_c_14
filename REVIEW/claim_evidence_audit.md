# Claim → evidence audit

Numbers in the manuscript are not typed: each is an expression evaluated by `manuscript/build_manuscript.py`
against `results/statistics.json`, `results/metrics.json` or `results/raw_results.csv`. Three headline numbers
and both propositions were additionally recomputed independently from the raw CSV / by direct calculation
(script in the session log; values matched: +4.24 points wide vs default at 40 % asymmetric noise, +2.20 / +1.26 /
−0.31 points LOO-EM vs wide ridge on the 60 confirmation datasets, 98 % of runs at the top of the default grid).

| Claim | Evidence | Experiment | Where |
|---|---|---|---|
| Default-grid MiniRocket loses accuracy steeply under label noise | mean accuracy per condition | main, 108 datasets × 5 seeds | Table 1, Fig. 3 |
| The default penalty grid is binding under noise | share of runs selecting the grid's upper end; distribution of selected penalty | main | Fig. 4, Sect. 6.1 |
| Widening the grid raises accuracy under noise and not on clean data | paired differences, Wilcoxon/Holm, CI, W/T/L | main | Table 2 |
| …but not for n < 100 at 30 % symmetric noise, nor for ≥ 10 classes at 40 % asymmetric | stratified means | main | Table 5, Sect. 6.1 |
| …and under symmetric noise not on balanced accuracy / macro-F1 | paired differences on both metrics; imbalance strata | main | Sect. 6.1, supplement |
| Symmetric noise: expected ridge decision function keeps its arg-max; variance controlled by the penalty | Proposition 1 (proof; unit test of the affine identity) | — | Sect. 3.2 |
| Asymmetric noise can move the arg-max | Proposition 2 with numerical counter-example (recomputed) | — | Sect. 3.2 |
| Cleaning baselines lower accuracy on clean labels | paired tests vs wide ridge | main + calibrated-CL pass | Table 1, Sect. 6.2, supplement |
| Calibration matters for confident learning | paired test calibrated vs uncalibrated | calibrated-CL pass | Sect. 6.2, supplement |
| LOO scores detect label errors as well as 5-fold scores | detection AUROC | main | Fig. 5, supplement |
| LOO-EM improves on wide-grid ridge at 30 % symmetric / 40 % asymmetric noise, costs ≈ 0.3 points on clean labels | paired tests on the confirmation set | main + BIC pass | Table 3 |
| LOO-EM's gains hold on balanced accuracy and macro-F1 | paired differences | main | Sect. 6.3 |
| LOO-EM ≈ calibrated confident learning at ≥ 20 % noise, better on clean / 10 % | paired tests (confirmation set) | main | Sect. 6.3 |
| LOO scores, the noise model and the wide grid are each necessary; soft targets are not | ablation variants | main + BIC pass | Table 4, Fig. 8 |
| v1 fails on clean many-class data; identifiability trade-off | v1 rows in all tables; class strata; BIC acceptance rates; penalty ×0.5 / ×2 | main | Sect. 6.4, Table 5 |
| No gain under boundary noise | paired tests | main | Sect. 6.5 |
| Findings transfer to Hydra features / multivariate data / larger sets | secondary runs (seeds 0–2) | Hydra (30 datasets), UEA (18), large (3) | Table 6 |
| Relation to QUANT, 1-NN, FCN baselines | secondary runs | families (108), deep (6) | Tables 7–8 |
| Cost: LOO-EM is a small multiple of the eigendecomposition and cheaper than 5-fold confident learning | per-run timings | main | Table (cost), Fig. 9 |
| Insensitivity to EM iterations / prior strength; sensitivity to the BIC penalty | sensitivity runs | 30 datasets, seeds 0–2 | Sect. 6.8, supplement |

**Claims removed or weakened during the audit**

* "essentially free" → replaced by measured ratios.
* "balanced accuracy and macro-F1 lead to the same conclusions" → false for the grid under symmetric noise; replaced by the measured result.
* "cleaning never helps" → calibrated confident learning helps at 30–40 % symmetric noise; text changed.
* "arg-max preserved iff T symmetric" → only the "if" direction plus a counter-example.
* Any statement about MultiRocket or ElectricDevices → removed (experiments not completed).
* "state of the art", "outperforms" → not used.
