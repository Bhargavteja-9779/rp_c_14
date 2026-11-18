# Regularise before you relabel: label noise in random convolutional kernel time series classification

Code, raw results, tables, figures and manuscript sources for a study of training-label noise in
ROCKET-family time series classifiers (random convolutional kernel features + ridge classifier).

**What the study found (all numbers are generated from `results/`):**

1. *Theory.* For a ridge classifier, symmetric label noise leaves the expected decision function order-preserving and
   only adds variance; label smoothing therefore changes no prediction; asymmetric noise can move the arg-max.
2. *Synthetic noise (108 UCR datasets).* The customary ROCKET penalty grid `logspace(-3, 3, 10)` saturates under
   noise; widening it raises accuracy substantially, but with the squared-error LOO criterion the gain under symmetric
   noise is bought with minority-class recall. **LOO-EM** (EM on closed-form leave-one-out predictions, BIC choice of
   the noise model) adds a modest gain under heavy class-conditional noise. Generic cleaning costs accuracy at low noise.
3. *Real annotation errors (8,528 PhysioNet/CinC 2017 ECGs, original vs expert-revised labels).* Training on the noisy
   labels costs well under one point; **no relabelling or removal method — LOO-EM included — improves agreement with the
   expert labels, and most reduce it**; label-error rankings recover the documented errors only modestly. Conclusions
   from synthetic noise did not transfer.
4. *Exploratory:* choosing the penalty with a leave-one-out criterion that matches the target metric (e.g. balanced
   error) is free and removes the minority-class penalty under symmetric noise.
5. Negative results are reported, not hidden: a failed first version of LOO-EM, a rejected instance-dependent noise
   model, an unstable backward-correction configuration, and the real-noise result above (`supplementary/protocol.md`).

## Layout

```
config/default.yaml        frozen protocol: seeds, penalty grids, noise models, every method variant
src/data/ucr.py            dataset download (via aeon) and stratified resampling
src/data/cinc2017.py       PhysioNet/CinC 2017 ECG recordings with original (v0) and expert-revised (v3) labels
src/preprocessing/         MiniRocket / MultiRocket / Hydra transforms with caching
src/models/ridge_loo.py    Gram ridge classifier with closed-form leave-one-out predictions
src/models/cleaning.py     LOO-EM (v1, BIC, rejected instance model), LOO relabel/remove, confident learning,
                           k-fold relabel, ENN, backward-corrected ridge, LOO-error penalty criteria
src/models/ctw.py          re-implementation of CTW (Ma et al., IJCAI 2023) following the official code
src/models/noise.py        symmetric, asymmetric (pair-flip) and boundary (instance-dependent) noise
src/training/runner.py     resumable experiment runner (one CSV per dataset)
src/evaluation/            metrics, aggregation, automatic table generation
src/statistics/tests.py    Wilcoxon signed-rank, Holm, rank-biserial effect size, bootstrap CI, Friedman ranks
src/visualization/plots.py every figure
experiments/               entry points for each experiment
scripts/                   dataset summary, leakage audit, citation check, journal timing measurements
tests/                     unit tests (closed-form LOO vs brute force, scikit-learn equivalence, noise models, ...)
results/                   raw per-run CSVs, raw_results.csv, metrics.json, statistics.json, logs
tables/  figures/          machine-generated
manuscript/                template, builder, references, generated Markdown / DOCX / PDF
supplementary/             protocol (incl. design history), design-stage results, supplementary tables
REVIEW/                    ten internal review rounds, citation and claim audits, final risk report
Target_Journal_Submission/ packaged submission
```

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

or `conda env create -f environment.yml`. Python 3.11–3.13. No GPU is needed; the device (CPU / CUDA / Apple MPS)
is detected automatically and only matters for the optional deep baselines. Building the PDF additionally needs
`pandoc` and LibreOffice **with the Math component** (`libreoffice-math`), otherwise equations are missing from the PDF
(the DOCX is unaffected).

## Data

Nothing has to be downloaded by hand. The UCR/UEA archives are fetched per dataset from
<https://www.timeseriesclassification.com> through `aeon` on first use and stored in `data/raw/` (about 0.8 GB).
The ECG data (100 MB) and the four label versions are fetched from PhysioNet
(<https://physionet.org/content/challenge-2017/1.0.0/>, Open Data Commons Attribution License v1.0) on first use.
`data/dataset_summary.csv` lists the 112 equal-length univariate problems; `tables/table_leakage_audit.csv` lists
exact and near duplicates across the archive's train/test splits.

## Reproducing

```bash
python run_all.py --mode quick     # ~2 minutes: unit tests + 6 small datasets, 2 seeds, tables and figures in results/quick/
python run_all.py --mode full      # the complete study (resumable; see cost below)
python run_all.py --mode report    # re-aggregate existing raw results -> results/*.json, tables/, figures/
python manuscript/build_manuscript.py   # manuscript Markdown/DOCX/PDF with every number injected from the results
python -m pytest -q tests
```

`--mode full` runs every experiment in the paper, each writing one result file per dataset/fold under
`results/raw/<stage>/` and skipping files that already exist (re-issue the same command to resume):
main benchmark (108 datasets x 5 seeds x 12 noise conditions), the BIC, calibrated-confident-learning, backward-correction/
instance-model and penalty-criterion passes, the class-wise/calibration pass, the three large datasets, Hydra features (fixed
30-dataset subset), QUANT and 1-NN, the sensitivity study, the multivariate UEA study, deep baselines, **the PhysioNet/CinC 2017
real-noise experiment** (`experiments/run_cinc2017.py`, `run_cinc2017_criteria.py`), **CTW** (`run_ctw.py`), the exact-LOO
sensitivity (`run_exact_loo.py`) and the scaling study (`run_scaling.py`), then the report. On the 15 GB machine used,
MultiRocket and ElectricDevices with the 5-fold baselines could not be completed and are not reported.

**Cost.** On the 4-core, 15 GB machine used for the paper the main experiment took roughly 5 hours and everything
else roughly another 10–15 hours. The feature cache (`data/cache/`) needs about 6 GB per seed for MiniRocket; delete
it at any time. Memory: keep at most 3 workers for MultiRocket.

**Determinism.** Seeds 0–4 are fixed in `config/default.yaml`; seed 0 is the archive split and seeds 1–4 are
stratified resamples. Each noise realisation is a deterministic function of (dataset, seed, noise model, rate).
Accuracies are reproducible exactly on the same library versions; timings are not.

## Using the method on your own data

```python
import numpy as np
from aeon.transformations.collection.convolution_based import MiniRocket
from src.models.ridge_loo import GramRidge
from src.models import cleaning

t = MiniRocket(n_kernels=10000, random_state=0)
F_train, F_test = t.fit_transform(X_train), t.transform(X_test)      # X: (n, channels, length)
C = int(y_train.max()) + 1                                           # labels 0..C-1
g = GramRidge(alphas=np.logspace(-3, 9, 25)).fit(F_train)            # one eigendecomposition
res = cleaning.loo_em_bic(g, y_train, C)                             # or cleaning.none(g, y_train, C) for plain ridge
scores, alpha = cleaning.final_predict(g, F_test, g.test_kernel(F_test), res, C)
y_pred = scores.argmax(1)
print(res["info"]["model"], res["info"].get("noise_rate"))            # chosen noise model, estimated rate
```

## Where each manuscript number comes from

`manuscript/manuscript_template.md` contains expressions such as `<<DPTS('loo_em_bic','ridge_wide','asym 40%','confirm')>>`
that `build_manuscript.py` evaluates against `results/statistics.json`, `results/metrics.json` and
`results/raw_results.csv`. Tables are rendered from the same files. The build fails loudly on an unresolved
expression, an uncited table/figure or a citation without a reference entry.

## Licence and citation

Code: MIT (see `LICENSE`). The datasets remain under the terms of the UCR/UEA archive and their original donors;
please cite the archive papers when using them.
