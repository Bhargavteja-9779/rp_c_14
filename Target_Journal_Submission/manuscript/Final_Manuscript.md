---
title: "Regularise before you relabel: random convolutional kernel time series classification under synthetic and real annotation noise"
author: "[Author One]^1\\*^, [Author Two]^2^"
---

^1^ [Department, Institution, City, Country] · ^2^ [Department, Institution, City, Country]

\* Corresponding author: [name], [e-mail address], ORCID [0000-0000-0000-0000]

## Abstract

Time series classifiers are often trained on labels from technicians, outsourced annotators or automatic rules, some of which are wrong. Random convolutional kernel classifiers (ROCKET, MiniRocket and successors) pair a fixed random-kernel transform with a ridge classifier and are widely used for their speed and accuracy, but their behaviour under label noise had not been studied, and noisy-label methods are usually evaluated with synthetic noise only. We prove that symmetric noise leaves the expected ridge decision function order-preserving and only inflates its variance, so that label smoothing changes nothing, and we propose LOO-EM, an expectation–maximisation procedure for class-conditional noise built on closed-form leave-one-out predictions and a single eigendecomposition. On 108 UCR datasets with synthetic noise, a wider penalty grid raises accuracy by up to 4.2 points and LOO-EM adds 2.2 points at 40% asymmetric noise on 60 held-out datasets. On real annotation errors in 8,528 ECG recordings whose original labels were later revised by experts, training on the noisy labels costs only 0.6 points, LOO-EM lowers agreement with the expert labels by 1.5 points, no relabelling or removal method helps, and error rankings recover the documented errors only modestly (AUROC 0.64–0.70). Selecting the penalty with the target metric in mind is the safest remedy; automatic relabelling should be validated on real noise before use.

**Keywords** Time series classification · Label noise · Random convolutional kernels · Ridge regression · Leave-one-out cross-validation · Expectation–maximisation

# 1 Introduction

Time series classification (TSC) underlies activity recognition, condition monitoring and physiological signal analysis. Among the many algorithm families that have been benchmarked on the UCR archive (Dau et al. 2019; Bagnall et al. 2017; Middlehurst et al. 2024a), methods that transform each series with a large number of random convolutional kernels and train a linear ridge classifier on the pooled outputs occupy a distinctive position: ROCKET (Dempster et al. 2020), MiniRocket (Dempster et al. 2021), MultiRocket (Tan et al. 2022) and Hydra (Dempster et al. 2023) approach the accuracy of much more expensive ensembles and deep networks (Ismail Fawaz et al. 2019; Ismail Fawaz et al. 2020) while training in seconds on a CPU. This makes them a natural default in applied work.

Applied labels are rarely perfect. Sensor recordings are annotated by fatigued operators, by weak heuristics, by outsourced annotation services, or by propagating a label over a whole recording session, and the resulting errors can be random or systematic (Frénay and Verleysen 2014; Atkinson and Metsis 2020). A well-documented case is the 2017 PhysioNet/Computing in Cardiology Challenge: its 8,528 training ECGs were labelled by an outsourced company with only about 10% over-read by experts, the labels were revised in several steps during the competition, and experts re-reading contentious recordings agreed with each other only fairly (Fleiss κ ≈ 0.25; Clifford et al. 2017). Research on label noise in TSC has concentrated on deep networks: self-supervised and ensembling schemes (Castellani et al. 2021), confident-sample augmentation (Ma et al. 2023) and multi-scale co-teaching (Liu et al. 2023), with distance-based alternatives appearing only recently (Li et al. 2026). For the random-kernel family we found no systematic study, although its training regime is unusual: roughly ten thousand features are fitted to training sets that often contain fewer than a few hundred series, so the ridge classifier operates close to interpolation, the regime in which fitting noise can be either benign or harmful depending on the regularisation (Bartlett et al. 2020).

This paper asks five questions, each tied to an experiment.

**RQ1.** How much accuracy do random-kernel ridge classifiers lose under symmetric, asymmetric and instance-dependent label noise, and how much of the loss is caused by the regularisation protocol rather than by the noise itself?

**RQ2.** Do generic label-cleaning strategies (confident learning, nearest-neighbour editing, relabelling from out-of-sample predictions) help when applied to these features?

**RQ3.** Can label correction be made cheap by exploiting the closed-form leave-one-out (LOO) predictions of ridge regression, and does it improve on a properly regularised ridge classifier?

**RQ4.** Do the answers hold for other transforms, for multivariate data, for larger training sets and relative to other classifier families, including a time-series-specific noisy-label method?

**RQ5.** Do conclusions drawn from synthetic noise survive real annotation errors?

Our contributions are the following.

1. *Theory.* For ridge classification, symmetric label noise scales the expected decision function without changing its arg-max and adds variance that only the penalty controls (Proposition 1); label smoothing is therefore a no-op (Corollary 1); asymmetric noise can move the arg-max (Proposition 2).
2. *A large-scale characterisation with synthetic noise.* MiniRocket with ridge regression on 108 UCR datasets, five resamples and twelve noise conditions (169,965 recorded runs of 27 method variants), with secondary experiments on Hydra features, 18 multivariate datasets, large training sets, class-wise and calibration metrics, and comparisons with QUANT, nearest neighbours, backward loss correction, deep baselines and a re-implementation of the time-series-specific method CTW.
3. *LOO-EM.* An EM algorithm for class-conditional noise in which the predictive distribution is always the closed-form LOO one, every iteration reuses one eigendecomposition (dual or primal), and BIC chooses between no noise model, a symmetric and a full transition matrix. A first version that failed on many-class data and a rejected instance-dependent extension are reported with their protocols.
4. *A real-noise evaluation.* On 8,528 ECG recordings with original operational labels and expert-revised labels, under a protocol fixed in advance, we show that the ranking of remedies obtained with synthetic noise does not transfer: label correction, LOO-EM included, does not help and mostly hurts, and the documented errors are only modestly detectable.
5. *Practical guidance*, including the observation that the leave-one-out criterion used to select the penalty, not the grid alone, decides whether regularisation protects minority classes, an end-to-end deployment profile, and a scaling analysis.

# 2 Related work

**Random convolutional kernel classifiers.** ROCKET convolves each series with 10,000 random kernels and pools each output by its maximum and its proportion of positive values (Dempster et al. 2020). MiniRocket fixes most of the randomness and keeps only the proportion-of-positive-values pooling (Dempster et al. 2021); MultiRocket adds first differences and three further pooling operators (Tan et al. 2022); Hydra counts which kernel in each group responds most strongly (Dempster et al. 2023). All of them train a ridge classifier whose penalty is chosen by leave-one-out cross-validation over ten values between 10^−3^ and 10^3^, and this grid is the default in widely used implementations (Middlehurst et al. 2024b). Recent benchmark studies place these methods among the most accurate fast classifiers (Middlehurst et al. 2024a); QUANT obtains comparable accuracy from interval quantiles and extremely randomised trees (Dempster et al. 2024a). The closest methodological relative of our work is prevalidated ridge regression (Dempster et al. 2024b), which uses closed-form LOO predictions, in the spirit of pre-validation (Tibshirani and Efron 2002), to scale ridge outputs into calibrated probabilities. It targets probability calibration on clean labels; it does not address label noise.

**Learning with label noise.** Learning from randomly mislabelled examples has been studied since Angluin and Laird (1988). Surveys distinguish noise that is independent of the features given the class (symmetric or asymmetric class-conditional noise) from instance-dependent noise (Frénay and Verleysen 2014; Song et al. 2023). Three families of remedies are relevant here. *Noise-robust losses and loss correction* modify the objective, for example through the generalised cross-entropy (Zhang and Sabuncu 2018), loss conditions for robustness (Ghosh et al. 2017), importance reweighting (Liu and Tao 2016) or forward and backward correction with an estimated transition matrix (Natarajan et al. 2013; Patrini et al. 2017). *Latent-label models* treat the true label as a hidden variable and estimate a transition matrix by EM, from the annotator model of Dawid and Skene (1979) to noise-robust logistic regression (Bootkrajang and Kabán 2012). *Data cleaning* identifies and removes or relabels suspicious samples, from edited nearest neighbours (Wilson 1972) and classification filters (Brodley and Friedl 1999) to confident learning (Northcutt et al. 2021), and sample-selection schemes such as Co-teaching (Han et al. 2018). Chen et al. (2021) showed that under symmetric noise accuracy measured on noisy validation labels is an affine function of clean accuracy, which justifies model selection on noisy labels; our Proposition 1 is the analogous statement for the ridge decision function itself.

**Label noise in time series.** Atkinson and Metsis (2020) studied the identification of mislabelled sensor segments. Castellani et al. (2021), Ma et al. (2023) and Liu et al. (2023) proposed deep networks for noisy-label TSC evaluated on subsets of the UCR and UEA archives: SREA combines self-supervised reconstruction with label correction, CTW applies time-warping augmentation only to samples selected by a class-normalised small-loss rule, and Scale-teaching exchanges clean-sample selections across temporal scales. More recent work extends these ideas with self-supervised pre-training and soft-label refurbishment (Liu et al. 2026), studies noise whose rate changes over time within a sequence (Nagaraj et al. 2025), addresses noisy labels in clinical time series with label augmentation and calibration (Li et al. 2025), and handles inconsistent labels between neighbouring segments (Chen et al. 2024). Li et al. (2026) proposed a granular-ball method built on dynamic time warping. A scoping review of noisy labels in medical prediction finds that most methods are evaluated with synthetic noise only (Wei et al. 2024). Real annotation errors in physiological data are documented for the 2017 PhysioNet/Computing in Cardiology Challenge, whose labels were revised by experts during the competition (Clifford et al. 2017) and in which one competing team, after a manual review, relabelled 197 of the 8,528 training recordings (Teijeiro et al. 2017). None of these works evaluates random-kernel classifiers, examines the role of the ridge penalty, or uses the closed-form LOO structure.

**Research gap.** What exists is (i) fast kernel-transform classifiers evaluated on clean benchmarks, (ii) generic and deep-learning-specific noise remedies evaluated mostly with synthetic noise, and (iii) a closed-form LOO calibration method for clean data. What is missing is knowledge of how the kernel-transform family behaves under label noise, whether generic or time-series-specific remedies help it, whether its algebraic structure can be exploited, and whether conclusions drawn from synthetic noise survive real annotation errors. This matters because these classifiers are used precisely where compute is scarce and labels are cheap, and therefore noisy.

# 3 Ridge classification on random kernel features under label noise

## 3.1 Model and closed-form leave-one-out predictions

Let $\phi(x)\in\mathbb{R}^p$ be the transform of a series $x$, and let $Z\in\mathbb{R}^{n\times p}$ hold the training features after each column has been standardised with training statistics and centred. Labels $y_i\in\{1,\dots,C\}$ are encoded as a matrix $Y\in\{-1,+1\}^{n\times C}$ with $Y_{ic}=+1$ iff $y_i=c$. The ridge classifier (Hoerl and Kennard 1970) with unpenalised intercept and penalty $\alpha$ predicts $\arg\max_c f_c(x)$ with

$$f(x)=\bar y+k(x)^\top (K+\alpha I)^{-1}(Y-\mathbf{1}\bar y^\top),\qquad K=ZZ^\top,\quad k(x)=Z\,z(x), \qquad\text{(1)}$$

where $\bar y$ is the vector of column means of $Y$. With the eigendecomposition $K=U\,\text{diag}(s)\,U^\top$ the fitted training values are $HY$ with the hat matrix

$$H(\alpha)=\tfrac1n\mathbf{1}\mathbf{1}^\top+U\,\text{diag}\!\left(\frac{s_j}{s_j+\alpha}\right)U^\top . \qquad\text{(2)}$$

The prediction for sample $i$ of the model trained without sample $i$ is available without refitting (Golub et al. 1979; Hastie et al. 2009):

$$F^{\text{loo}}_{i\cdot}=\frac{(HY)_{i\cdot}-h_{ii}Y_{i\cdot}}{1-h_{ii}},\qquad\text{i.e.}\quad F^{\text{loo}}=M(\alpha)\,Y,\quad M=D^{-1}\big(H-\text{diag}(h)\big),\ D=\text{diag}(1-h_{ii}). \qquad\text{(3)}$$

Two properties of Eq. (3) drive the rest of the paper. (a) $M(\alpha)$ has a zero diagonal: the LOO score of a sample does not depend on its own label. (b) $M(\alpha)$ does not depend on $Y$ at all: after any edit of the labels the complete set of LOO scores is recomputed by one $n\times n$ by $n\times C$ product. As in standard pipelines the column scaling is computed once on all training series and is not recomputed inside the LOO formula.

## 3.2 Symmetric noise adds variance, asymmetric noise adds bias

Suppose the observed labels $\tilde y_i$ are drawn from the true labels by a transition matrix $T$ with $T_{zk}=P(\tilde y=k\mid y=z)$, independently of $x$ given $y$. Write $\tilde Y$ for the encoded noisy labels and $g(x)=(f(x)+\mathbf 1)/2$ for the clean decision function on the 0/1 scale; because of the intercept the entries of $g(x)$ sum to one.

**Proposition 1** (symmetric noise). *Let $T=(1-\rho)I+\frac{\rho}{C-1}(\mathbf 1\mathbf 1^\top-I)$ with $\rho<(C-1)/C$. For every $x$ and every penalty $\alpha$, the decision function $\tilde f$ trained on $\tilde Y$ satisfies*

$$\mathbb{E}\big[\tilde f(x)\mid Y\big]=a\,f(x)+b\,\mathbf 1,\qquad a=1-\frac{\rho C}{C-1}>0,\quad b=\frac{\rho(2-C)}{C-1}, \qquad\text{(4)}$$

*so the arg-max of the expected decision function equals that of the clean one. Moreover $\tilde f_c(x)-\mathbb{E}[\tilde f_c(x)\mid Y]=\sum_i w_i(x;\alpha)\,\varepsilon_{ic}$ with $w(x;\alpha)^\top=\tfrac1n\mathbf 1^\top+k(x)^\top(K+\alpha I)^{-1}(I-\tfrac1n\mathbf 1\mathbf 1^\top)$ and zero-mean $\varepsilon_{ic}$, independent across samples $i$, of variance at most 1; the non-constant part of $w$ shrinks monotonically to zero as $\alpha$ grows.*

*Proof.* Entry-wise, $\mathbb{E}[\tilde Y_{ic}\mid Y]$ equals $1-2\rho$ if $Y_{ic}=1$ and $-1+2\rho/(C-1)$ otherwise, i.e. $\mathbb{E}[\tilde Y\mid Y]=aY+b\mathbf 1\mathbf 1^\top$. Equation (1) is linear in the targets and reproduces constants through the intercept, which gives Eq. (4). The second statement is Eq. (1) applied to $\tilde Y-\mathbb{E}[\tilde Y\mid Y]$, whose rows are independent with bounded entries; the eigenvalues of $(K+\alpha I)^{-1}$ decrease in $\alpha$. $\square$

**Corollary 1** (label smoothing). *Replacing one-hot targets by $(1-\varepsilon)$ one-hot $+\,\varepsilon/C$, a common remedy for noisy labels (Lukasik et al. 2020), maps $Y$ affinely as in Eq. (4) and therefore changes no prediction of a ridge classifier with intercept, for any $\varepsilon<1$ and any penalty.* On the $\pm1$ scale the smoothed targets are $(1-\varepsilon)Y+\varepsilon(2/C-1)\mathbf 1\mathbf 1^\top$, the form of Eq. (4) with $\rho=\varepsilon(C-1)/C$; the slope $1-\varepsilon$ is positive and the constant is absorbed by the intercept. Label smoothing is therefore not evaluated as a separate baseline.

Symmetric noise therefore attenuates the signal by the factor $a$ and adds a variance term governed by $\|w(x;\alpha)\|$. The penalty that balances the two grows with $\rho$; a grid that is adequate for clean labels can become binding. This is the testable prediction behind RQ1.

**Proposition 2** (class-conditional noise in general). *For an arbitrary transition matrix, $\mathbb{E}[\tilde g(x)\mid Y]=T^\top g(x)$. The arg-max is preserved for all $g$ on the simplex if $T=aI+b\,\mathbf 1\mathbf 1^\top$ with $a>0$, which is the symmetric model; for other $T$ it need not be.*

*Proof.* On the 0/1 scale $\mathbb{E}[\tilde P\mid P]=PT$ and Eq. (1) is linear, so $\mathbb{E}[\tilde g(x)]^\top=g(x)^\top T$. If $T=aI+b\mathbf 1\mathbf 1^\top$, then $T^\top g=a\,g+b\,\mathbf 1$ because the entries of $g$ sum to one. For a counter-example with pair-flip noise ($T_{c,c}=1-\rho$, $T_{c,c+1}=\rho$, indices modulo $C$) take $C=3$, $\rho=0.3$ and $g=(0.45,0.40,0.15)$: then $T^\top g=(0.36,0.415,0.225)$ and the arg-max moves from the first to the second class. $\square$

Under asymmetric noise the expected decision function is thus wrong for points whose clean scores are not close to a vertex, however the penalty is chosen. Removing this bias requires a model of $T$, which motivates Sect. 4.

# 4 LOO-EM: expectation–maximisation on leave-one-out predictions

## 4.1 Latent-label model

We treat the true label $z_i$ as latent, with the usual class-conditional observation model $P(\tilde y_i=k\mid z_i=z,x_i)=T_{zk}$ (Dawid and Skene 1979; Bootkrajang and Kabán 2012). A classifier that is refitted on its own posteriors and then asked to score the same training points confirms itself; with $p\gg n$ a nearly interpolating ridge fit reproduces whatever targets it is given. We therefore let the predictive distribution for sample $i$ always be the leave-one-out one. Let $Q\in[0,1]^{n\times C}$ hold the current posteriors $q_i(z)$, initialised to the one-hot observed labels. One iteration consists of:

*M-step (classifier).* Fit the ridge model to the soft targets $2Q-1$ and obtain $F^{\text{loo}}=M(\alpha)(2Q-1)$ by Eq. (3), with $\alpha$ re-selected by the LOO squared error on the current targets.

*M-step (noise model).* With $p_i(z)=\text{softmax}(\beta F^{\text{loo}}_{i\cdot})_z$, maximise the observed-label LOO log-likelihood

$$\ell(\beta,T)=\sum_{i=1}^{n}\log\sum_{z=1}^{C}p_i(z)\,T_{z\tilde y_i} \qquad\text{(5)}$$

by alternating a one-dimensional search over the inverse temperature $\beta$ and the closed-form update $T_{zk}\propto N_{zk}+m\,T^{0}_{zk}$, where $N_{zk}=\sum_i r_i(z)\mathbb{1}[\tilde y_i=k]$, $r_i(z)\propto p_i(z)T_{z\tilde y_i}$, and $T^0$ is a symmetric matrix with the current overall noise rate (prior strength $m=C$). For the symmetric model only the rate is estimated.

*E-step.* $q_i(z)\propto p_i(z)\,T_{z\tilde y_i}$.

The loop stops when the largest change in $Q$ is below $10^{-3}$ or after 20 iterations. The final classifier is the ridge fit on the converged soft targets.

## 4.2 Choosing the noise model

A full transition matrix has $C(C-1)$ free parameters and can absorb ordinary classification error as "noise". We fit the symmetric model ($k=1$) and the full model ($k=C(C-1)$) and compare them with the noise-free model ($k=0$, $T=I$, $\beta$ fitted on the first-round LOO scores) by the Bayesian information criterion (Schwarz 1978) applied to Eq. (5), $\ell-\tfrac{k}{2}\log n$. If the noise-free model wins, the labels are left untouched and the method reduces to wide-grid ridge. Fig. 1 summarises the procedure.

![**Fig. 1** Overview of LOO-EM. (a) The transform, one eigendecomposition and the leave-one-out operators $M(\alpha)$ do not depend on the labels and are computed once. (b) Each EM iteration alternates a classifier M-step (ridge on the current soft targets, evaluated by its LOO predictions), a noise-model M-step (temperature and transition matrix by maximum LOO likelihood) and an E-step (posterior over the true label). (c) BIC chooses between no noise model, a symmetric and a full transition matrix; the outputs are the final classifier, calibrated probabilities and a ranked list of suspicious labels](/home/user/rp_c_14/figures/fig_method_overview.png){width=6.3in}

## 4.3 Cost

The eigendecomposition is needed by any LOO-tuned ridge classifier. When $n\le p$ we decompose the $n\times n$ Gram matrix ($O(n^2p+n^3)$ time, $O(n^2)$ memory); when $n>p$ we decompose the $p\times p$ scatter matrix $Z^\top Z=V\,\text{diag}(s)V^\top$ and set $U=ZVS^{-1/2}$ ($O(np^2+p^3)$ time, $O(np)$ memory). Both routes yield the same thin factor $U$ and eigenvalues, so Eqs. (2)–(3) and all later steps are unchanged (verified numerically in the unit tests). Each EM iteration costs $O(n^2C)$ per candidate penalty; the hat diagonal for each penalty is label-independent and cached, and after the first iteration only penalties adjacent to the current one are examined, extending the window while the optimum lies on its edge. No model is ever refitted from scratch. In contrast, k-fold cleaning needs $k$ further eigendecompositions.

## 4.4 What the likelihood cannot decide

Equation (5) cannot by itself distinguish heavy class-conditional noise from genuinely overlapping classes: a clean-posterior model with large off-diagonal $T$ and a flat predictor with $T=I$ can reach the same likelihood. Any procedure that corrects aggressively when predictions and labels disagree will therefore also "correct" hard but correctly labelled data. BIC resolves the ambiguity conservatively on small training sets, and Sect. 6.4 quantifies the price of that choice. It cannot do so on large ones: the BIC penalty grows like $\log n$, whereas the likelihood gained by letting $T$ absorb a systematic confusion of the classifier grows like $n$, so with enough data any consistent confusion will be accepted as noise. Section 6.6 shows this on real data.

## 4.5 From the model to practice

The procedure leaves three decisions to the user, and the experiments in Sect. 6 inform each. (i) *Regularisation.* A grid that is wide enough not to saturate costs nothing, but the criterion that picks the penalty from it matters: the leave-one-out squared error, the usual default, favours majority classes, which on the UCR data cost minority-class recall under symmetric noise and on the ECG data cost accuracy as well (Sects. 6.1, 6.5–6.7). The criterion should be chosen to match the evaluation metric. (ii) *Label correction* is applied only when BIC prefers a noise model; when it does not, LOO-EM returns the unmodified labels and coincides with wide-grid ridge. The chosen model and the estimated noise rate are reported to the user, and a low estimated rate is a signal that correction is unlikely to pay off (Sects. 6.3 and 6.6). (iii) *Expert review*: the posterior $1-q_i(\tilde y_i)$ ranks the training labels by suspicion. Because the ranking is a by-product of the fit, a practitioner can send the top of the list for re-annotation instead of, or in addition to, accepting automatic corrections. Section 6.6 measures how well this ranking recovers real, expert-documented annotation errors.

# 5 Experimental setup

**Data.** The 112 equal-length univariate problems of the UCR archive (Dau et al. 2019) were obtained through the aeon toolkit (Middlehurst et al. 2024b). The 108 problems with fewer than 3,000 training series form the main benchmark (training sizes 16–1,800, median 180; 2–60 classes; Fig. 2); of the four larger ones, FordA, FordB and Crop (3,601–7,200 training series) are used for a separate large-data experiment without the 5-fold baselines, which are too expensive there, and ElectricDevices (16,637 series in total) for a scaling experiment with training sets of 500 to 14,000 series. Generalisation to multivariate data uses the equal-length UEA problems that fit the available machine (18 datasets).

**Real annotation noise.** The 8,528 public training recordings of the 2017 PhysioNet/Computing in Cardiology Challenge (Clifford et al. 2017; Goldberger et al. 2000; Pollard et al. 2026): single-lead ECGs of 9–61 s recorded by patients with a hand-held device, labelled normal rhythm (N), atrial fibrillation (A), other rhythm (O) or too noisy to classify (~). PhysioNet distributes four label files. The earliest (version 0) holds the labels released at the start of the challenge, which the challenge paper describes as produced by an outsourced annotation company with about 10% over-read by experts (Clifford et al. 2017; the dataset page describes them as single-expert labels). The later files record the revisions: version 1 relabels 237 recordings, all as too noisy to classify, after a visual re-check of every recording, and versions 2 and 3 change a further 205 labels after expert review of contentious recordings. In total 412 labels (4.8%) differ between versions 0 and 3 (Table 1). We train every method on the version-0 labels and evaluate against version 3, which we call the *expert-revised* labels: they are not error-free (inter-expert agreement on contentious recordings was low, Fleiss κ ≈ 0.25), so all results measure agreement with the revised reference and the detection analysis measures recovery of *documented* corrections. Recordings are z-normalised, decimated to 150 Hz and centre-cropped or tiled to 30 s. Evaluation uses stratified 5-fold cross-validation repeated twice (fold seeds 0 and 1); every transform, scaling statistic, penalty and EM quantity is fitted on training folds only. The protocol, including the primary comparisons and tests, was written before any method was run on these data (supplementary protocol).

|              | N (revised)   | A (revised)   | O (revised)   | ~ (revised)   | Total   |
|:-------------|:--------------|:--------------|:--------------|:--------------|:--------|
| N (original) | 5019          | 9             | 12            | 114           | 5154    |
| A (original) | 19            | 684           | 36            | 32            | 771     |
| O (original) | 38            | 65            | 2367          | 87            | 2557    |
| ~ (original) | 0             | 0             | 0             | 46            | 46      |
| Total        | 5076          | 758           | 2415          | 279           | 8528    |

: **Table 1** Original (version 0) against expert-revised (version 3) labels of the 8,528 ECG recordings; N normal, A atrial fibrillation, O other rhythm, ~ too noisy. Off-diagonal cells are the documented annotation errors


![**Fig. 2** Training-set size, number of classes and series length of the UCR datasets used](/home/user/rp_c_14/figures/fig_dataset_distribution.png){width=6.3in}

**Resampling and seeds.** Five seeds (0–4) were fixed in advance. Seed 0 is the archive split; seeds 1–4 are stratified resamples of the pooled data with the original training and test sizes. Secondary experiments use seeds 0–2. Per-dataset scores are means over seeds.

**Noise.** Noise is applied to training labels only; test labels are always clean. *Symmetric*: a fraction $\rho\in\{0,0.1,0.2,0.3,0.4\}$ of labels is replaced by a uniformly chosen different class. *Asymmetric*: a fraction $\rho\in\{0.1,\dots,0.4\}$ of each class $c$ is relabelled $c+1$ (mod $C$). *Boundary*: the fraction $\rho\in\{0.1,0.2,0.3\}$ of training series with the smallest clean LOO margin is relabelled to the strongest competing class; this instance-dependent model is adversarial to margin-based detectors by construction. Each noise realisation is a deterministic function of dataset, seed, noise model and rate, so all methods see identical labels. Conditions in which a class disappears from the noisy training set are skipped for that dataset and seed.

**Features and classifier.** MiniRocket with 10,000 kernels as implemented in aeon, fitted on training series only; columns standardised with training statistics; ridge classifier as in Sect. 3.1. Our implementation reproduces scikit-learn's `RidgeClassifierCV` (identical selected penalty and predictions in a unit test). The *default grid* is ten log-spaced penalties in [10^−3^, 10^3^]; the *wide grid* is 25 log-spaced penalties in [10^−3^, 10^9^].

**Methods compared on identical features and noise.** (i) Ridge with the default grid; (ii) ridge with the wide grid; (iii) LOO-EM with BIC selection (proposed); (iv) LOO-EM v1, the earlier frozen variant with a full transition matrix accepted through a one-parameter gate; (v) confident learning (Northcutt et al. 2021) using the `cleanlab` implementation with 5-fold out-of-sample ridge scores passed through a softmax whose temperature is fitted by maximum likelihood on those scores, flagged series removed (the uncalibrated variant with unit temperature, which we ran first, is kept in the tables because the difference is instructive); (vi) relabelling from the same 5-fold predictions; (vii) edited nearest neighbours with $k=3$ in the standardised feature space (Wilson 1972); (viii) one-round relabelling and (ix) removal of series whose LOO prediction disagrees with their label. Methods (iii)–(ix) use the wide grid. (x) Backward loss correction (Natarajan et al. 2013; Patrini et al. 2017) applied to the squared loss: ridge fitted to the corrected targets onehot($\tilde y$)$\hat T^{-1}$, which are unbiased for the clean targets, with $\hat T$ estimated by anchor points from calibrated LOO probabilities; of the two anchor rules described by Patrini et al. (97th percentile, arg-max) we use the one that was better on design data (arg-max). (xi) A time-series-specific noisy-label method, CTW (Ma et al. 2023), re-implemented from the official code (architecture, class-normalised small-loss selection with a ten-epoch loss memory, time-warping of selected samples, reconstruction loss, 300 epochs, the authors' hyper-parameters) and run on exactly our splits and noise realisations. Because 300 CPU epochs per run rule out the full benchmark, CTW is evaluated on the UCR datasets of the CTW paper's own experiment list; MelbournePedestrian is excluded because its series have unequal lengths and missing values and it is therefore not part of our benchmark, leaving seven datasets. The re-implementation was checked against published CTW accuracies (Sect. 6.8). Other classifier families trained on the same noisy labels: QUANT (Dempster et al. 2024a) and 1-nearest-neighbour with Euclidean distance, and, on six small datasets, a fully convolutional network (Wang et al. 2017) trained with cross-entropy, with the generalised cross-entropy loss (Zhang and Sabuncu 2018) and with Co-teaching (Han et al. 2018) given the true noise rate. Deviations from original implementations: the deep baselines are trained for 100 epochs with Adam (learning rate 10^−3^, batch size up to 32) without early stopping, since no clean validation labels exist.

**Protocol and design stages.** Method design used 12 development datasets drawn at random. The resulting configuration (v1) was frozen and run on all datasets. An interim inspection of the first completed datasets showed that v1 lost accuracy on clean, many-class data (Sect. 6.4). We then designed the BIC variant on a *design set* of 39 datasets and froze it before any other result was inspected. The 60 datasets that belong to neither the development nor the design set form the *confirmation set* on which the primary comparison is made. An anchor-point variant (Patrini et al. 2017) tried at the same stage was worse than wide-grid ridge on the design set and was dropped. In response to review we added, before running them, the backward-correction and CTW baselines, the ECG experiment, and a candidate instance-dependent noise model with a fixed adoption rule (stage 3, Sect. 6.4). An exploratory comparison of penalty-selection criteria, prompted by the ECG results, is reported separately and labelled as such (stage 4, Sect. 6.7). The full history is in the supplementary protocol.

**Metrics and statistics.** The primary metric is test accuracy. Balanced accuracy and macro-F1 are reported alongside it where they change a conclusion (Sect. 6.1) and in full in the supplement. For label-error detection we report the AUROC of each method's suspicion score and the fraction of training labels that are correct after cleaning. Class-wise behaviour is measured by the recall of the smallest training class and the worst per-class recall; probabilistic quality by the expected calibration error (15 equal-width bins of top-label confidence; Guo et al. 2017) and the Brier score (Brier 1950), with probabilities obtained by fitting one temperature to each final model's own LOO decision values against its training targets, so that no clean labels are used. Methods are compared by two-sided Wilcoxon signed-rank tests (Wilcoxon 1945) over datasets (unit of analysis: the dataset, as recommended by Demšar 2006), with Holm correction over the twelve noise conditions (Holm 1979), the matched-pairs rank-biserial correlation as effect size, and percentile bootstrap 95% confidence intervals (10,000 resamples of datasets) for the mean difference. The test is chosen because per-dataset accuracy differences are paired, bounded and not normally distributed. On the ECG data, where all methods are evaluated on the same recordings, we use McNemar's exact test on paired per-recording correctness (McNemar 1947), separately for each repetition, and a paired bootstrap over recordings (Efron 1979; 10,000 resamples for accuracy, 1,000 for the challenge F1) for confidence intervals; the challenge metric is the mean F1 of classes N, A and O. Holm correction is applied over the two pre-specified primary ECG comparisons.

**Leakage audit.** Transforms, scaling, penalty selection and all EM quantities use training data only. A duplicate scan of the archive splits found exact train–test duplicates in 16 datasets; 8 datasets have more than 1% duplicated test series (Crop, ElectricDevices, InlineSkate, InsectEPGSmallTrain, PowerCons, ToeSegmentation1, Worms, WormsTwoClass). This affects all methods equally; the main comparison is repeated without these datasets in the supplement and is unchanged in direction.

**Implementation.** Python 3.13, aeon 1.6, scikit-learn 1.9, NumPy/SciPy, cleanlab 2.9, PyTorch 2.15 (CPU). All experiments ran on a four-core CPU machine with 15 GB of memory; no GPU was used. Wall-clock times are from this shared machine and should be read as relative rather than absolute.

# 6 Results

## 6.1 RQ1: fragility, and how much of it is the penalty grid

Table 2 and Fig. 3 give mean test accuracy over the 108 datasets. With the default grid, accuracy falls from 87.1% on clean labels to 79.1% at 20% and 64.1% at 40% symmetric noise, and to 59.6% at 40% asymmetric noise.

| Method                            | S 0   | S 10   | S 20   | S 30   | S 40   | A 10   | A 20   | A 30   | A 40   | B 10   | B 20   | B 30   |
|:----------------------------------|:------|:-------|:-------|:-------|:-------|:-------|:-------|:-------|:-------|:-------|:-------|:-------|
| Ridge (default grid)              | 87.1  | 83.6   | 79.1   | 72.2   | 64.1   | 84.1   | 79.6   | 70.4   | 59.6   | 83.0   | 79.2   | 74.7   |
| Ridge (wide grid)                 | 87.2  | 83.9   | 79.6   | 73.6   | 65.8   | 84.5   | 81.3   | 73.5   | 63.8   | 83.2   | 80.1   | 76.9   |
| LOO-EM (BIC, proposed)            | 86.8  | 84.1   | 80.2   | 74.8   | 66.8   | 84.7   | 81.9   | 74.1   | 65.8   | 83.1   | 80.0   | 76.9   |
| LOO-EM v1 (full T)                | 85.4  | 82.4   | 78.6   | 73.5   | 65.8   | 82.7   | 80.2   | 74.0   | 67.4   | 82.8   | 79.9   | 76.4   |
| Confident learning (calibrated)   | 85.0  | 83.0   | 79.9   | 75.1   | 67.1   | 83.4   | 81.1   | 74.0   | 64.5   | 82.9   | 80.7   | 77.7   |
| Confident learning (uncalibrated) | 83.7  | 82.0   | 78.0   | 72.6   | 64.8   | 82.0   | 79.4   | 72.8   | 63.7   | 82.5   | 80.5   | 77.4   |
| Backward-corrected ridge          | 84.6  | 80.0   | 76.5   | 69.5   | 58.8   | 81.4   | 76.6   | 68.7   | 59.1   | 81.1   | 79.9   | 76.9   |
| 5-fold relabel                    | 83.2  | 80.6   | 76.3   | 70.8   | 63.8   | 80.8   | 78.0   | 71.4   | 63.1   | 82.1   | 80.1   | 77.4   |
| ENN                               | 80.5  | 78.9   | 76.8   | 72.6   | 64.9   | 79.1   | 76.9   | 70.8   | 60.4   | 79.4   | 79.5   | 76.3   |
| LOO relabel (1 round)             | 82.6  | 80.4   | 77.0   | 71.3   | 64.2   | 80.7   | 78.4   | 71.2   | 62.3   | 82.1   | 80.7   | 78.3   |
| LOO remove                        | 84.1  | 82.2   | 79.1   | 73.5   | 66.2   | 82.5   | 80.7   | 73.9   | 65.5   | 82.8   | 80.7   | 78.1   |

: **Table 2** Mean test accuracy (%) over 108 UCR datasets (five resamples each) on MiniRocket features. For each column only datasets on which the condition is defined for all methods are averaged. S, A and B denote symmetric, asymmetric and boundary noise at the given rate (%)


![**Fig. 3** Mean test accuracy over datasets as a function of the noise rate for the three noise models](/home/user/rp_c_14/figures/fig_accuracy_vs_noise.png){width=6.3in}

The selected penalty tells the story predicted by Proposition 1 (Fig. 4). On clean labels the default grid's upper end is selected in 34% of runs; at 20% symmetric noise in 96% and at 30% in 98%. With the wide grid, 95% of runs at 30% noise select a penalty above 10^3^.

![**Fig. 4** Distribution of the LOO-selected penalty under symmetric noise for the default and the wide grid](/home/user/rp_c_14/figures/fig_selected_alpha.png){width=6.3in}

Widening the grid leaves clean accuracy unchanged (+0.1 points, 95% CI −0.1 to +0.2, Holm-adjusted p = 0.648, 21 wins and 32 losses) and helps under every noise model (Table 3): +1.4 points, 95% CI +0.3 to +2.5, Holm-adjusted p = 0.011, 70 wins and 36 losses at 30% symmetric, +4.2 points, 95% CI +2.6 to +5.9, Holm-adjusted p < 0.001, 80 wins and 28 losses at 40% asymmetric and +2.2 points, 95% CI +1.5 to +3.0, Holm-adjusted p < 0.001, 71 wins and 17 losses at 30% boundary noise. The gain under asymmetric noise is larger than under symmetric noise, although regularisation cannot remove the bias of Proposition 2: it removes the variance component, which is present under every noise model. The benefit is not uniform. For the 36 datasets with fewer than 100 training series the wide grid is worse than the default at 30% symmetric noise (68.5% versus 70.5%), where the LOO criterion itself is computed from a few dozen noisy labels, and for the 21 datasets with ten or more classes it does not help at 40% asymmetric noise (53.8% versus 54.9%).

**The gain depends on the metric.** Accuracy rewards a classifier that drifts towards majority classes, and a large penalty shrinks the decision function towards the class-frequency intercept. Measured by balanced accuracy the wide grid changes performance under symmetric noise by −0.2, +0.4 and −0.1 points at 20%, 30% and 40%, and measured by macro-F1 by −0.6, −0.5 and −3.1 points, Holm-adjusted p = 0.028, 40 wins and 65 losses. The accuracy gain under symmetric noise is thus real only for roughly balanced problems: on the 59 datasets whose largest class is less than 1.5 times the smallest, balanced accuracy at 30% symmetric noise rises from 72.7% to 73.9%, whereas on the 22 datasets with a ratio of 3 or more accuracy rises from 70.1% to 73.8% while balanced accuracy does not move (61.8% and 61.6%). Under asymmetric and boundary noise the wide grid helps on every metric, by smaller amounts: +2.7 points, Holm-adjusted p < 0.001, 74 wins and 34 losses in balanced accuracy and +1.6 points, Holm-adjusted p < 0.001, 59 wins and 29 losses in macro-F1 at the heaviest rates. Proposition 1 concerns the arg-max of the expected decision function and is silent on which classes absorb the variance; the data say that it is the minority classes.

| Noise        | n   | Mean diff.   | 95% CI         | p (Wilcoxon)   | p (Holm)   | Effect (r)   | W/T/L    |
|:-------------|:----|:-------------|:---------------|:---------------|:-----------|:-------------|:---------|
| sym 0%       | 108 | +0.07        | -0.06 to +0.25 | 0.6484         | 0.6484     | -0.07        | 21/55/32 |
| sym 10%      | 108 | +0.26        | -0.03 to +0.55 | 0.0110         | 0.0438     | +0.30        | 60/15/33 |
| sym 20%      | 107 | +0.56        | -0.07 to +1.15 | 0.0073         | 0.0366     | +0.32        | 61/12/34 |
| sym 30%      | 106 | +1.44        | +0.34 to +2.51 | 0.0018         | 0.0110     | +0.35        | 70/0/36  |
| sym 40%      | 105 | +1.66        | +0.27 to +3.07 | 0.0547         | 0.1094     | +0.22        | 63/0/42  |
| asym 10%     | 108 | +0.34        | +0.07 to +0.62 | 0.0145         | 0.0438     | +0.30        | 53/22/33 |
| asym 20%     | 108 | +1.61        | +1.03 to +2.17 | <0.0001        | <0.0001    | +0.65        | 70/13/25 |
| asym 30%     | 108 | +3.08        | +1.69 to +4.36 | <0.0001        | <0.0001    | +0.63        | 84/2/22  |
| asym 40%     | 108 | +4.24        | +2.57 to +5.88 | <0.0001        | <0.0001    | +0.61        | 80/0/28  |
| boundary 10% | 105 | +0.15        | +0.04 to +0.26 | 0.0009         | 0.0065     | +0.46        | 48/35/22 |
| boundary 20% | 101 | +0.84        | +0.44 to +1.25 | <0.0001        | <0.0001    | +0.59        | 55/21/25 |
| boundary 30% | 102 | +2.24        | +1.55 to +2.95 | <0.0001        | <0.0001    | +0.76        | 71/14/17 |

: **Table 3** Wide versus default penalty grid, all 108 datasets: paired differences in accuracy points, two-sided Wilcoxon signed-rank test, Holm correction over the twelve conditions, matched-pairs rank-biserial effect size, and wins/ties/losses over datasets


## 6.2 RQ2: cleaning detects mislabelled series but lowers accuracy

Every cleaning baseline is worse than wide-grid ridge on clean labels (Table 2): calibrated confident learning by 2.1 points (11 wins and 74 losses), 5-fold relabelling by 4.0, LOO relabelling by 4.6, LOO removal by 3.1 and edited nearest neighbours by 6.7 points. At 10% symmetric or asymmetric noise all of them remain below wide-grid ridge on average. Calibrated confident learning becomes useful at high symmetric noise, +1.5 points, 95% CI +0.7 to +2.3, Holm-adjusted p = 0.001, 68 wins and 37 losses at 30% and +1.4 points, 95% CI +0.7 to +2.0, Holm-adjusted p < 0.001, 72 wins and 32 losses at 40%, but not significantly so under asymmetric noise (+0.6 points, 95% CI −0.4 to +1.6, Holm-adjusted p = 0.299, 65 wins and 43 losses at 40%). Under boundary noise one-round LOO relabelling is the best method and gains 1.4 points at the 30% rate.

How the baseline is fed matters. Our first implementation passed raw ridge scores through a unit-temperature softmax, which yields nearly flat probabilities; fitting a single temperature improves confident learning at every symmetric and asymmetric rate, by 1.3 points on clean labels and 2.5 points at 30% symmetric noise (both Holm-adjusted p < 0.001). The uncalibrated variant would have understated the baseline, and all comparisons below use the calibrated one.

The reason is not poor detection. The AUROC of the LOO suspicion score is 0.94 at 10% and 0.91 at 20% symmetric noise, the same as for 5-fold scores (0.94 and 0.91) at a fraction of the cost (Fig. 5). But every clean series that a classifier cannot predict from the others is also flagged, and on a dataset with clean LOO error $e$ a disagreement rule removes or flips roughly a fraction $e$ of correct labels. A regularised ridge classifier tolerates a noisy label better than it tolerates the loss, or the inversion, of a hard but correct one.

![**Fig. 5** AUROC of label-error detection for each method's suspicion score](/home/user/rp_c_14/figures/fig_detection_auroc.png){width=6.3in}

## 6.3 RQ3: LOO-EM against a well-regularised ridge classifier

On the confirmation set (Table 4) LOO-EM changes clean accuracy by −0.3 points, 95% CI −0.6 to −0.1, Holm-adjusted p = 0.051, 7 wins and 21 losses, and improves on wide-grid ridge by +1.3 points, 95% CI +0.6 to +2.0, Holm-adjusted p = 0.026, 36 wins and 21 losses at 30% symmetric noise and by +2.2 points, 95% CI +1.1 to +3.5, Holm-adjusted p = 0.012, 28 wins and 10 losses at 40% asymmetric noise. The remaining conditions show smaller mean gains (0.1–0.8 points for symmetric and asymmetric noise) that are not significant after Holm correction on 60 datasets, and no effect for boundary noise. Over all 108 datasets, which include the design data and are therefore optimistic for LOO-EM, the differences have the same signs and sizes: −0.3 points, 95% CI −0.6 to −0.2, Holm-adjusted p = 0.003, 13 wins and 35 losses on clean labels, +0.6 points, 95% CI +0.1 to +1.1, Holm-adjusted p = 0.022, 64 wins and 35 losses at 20% and +1.1 points, 95% CI +0.6 to +1.6, Holm-adjusted p < 0.001, 55 wins and 26 losses at 40% symmetric noise, and +2.0 points, 95% CI +1.2 to +2.9, Holm-adjusted p < 0.001, 50 wins and 17 losses at 40% asymmetric noise (Fig. 6). LOO-EM's gains do not depend on the metric: over all datasets it improves balanced accuracy over wide-grid ridge by +1.3 points, Holm-adjusted p < 0.001, 68 wins and 32 losses at 30% symmetric and +1.9 points, Holm-adjusted p < 0.001, 54 wins and 16 losses at 40% asymmetric noise, and macro-F1 by +1.2 and +1.8 points. Relative to the default pipeline the combined effect of the wide grid and LOO-EM on the confirmation set is +2.8 points, 95% CI +1.5 to +4.0, Holm-adjusted p = 0.003, 42 wins and 16 losses at 30% symmetric and +6.1 points, 95% CI +3.2 to +8.9, Holm-adjusted p < 0.001, 47 wins and 12 losses at 40% asymmetric noise.

| Noise        | n   | Mean diff.   | 95% CI         | p (Wilcoxon)   | p (Holm)   | Effect (r)   | W/T/L    |
|:-------------|:----|:-------------|:---------------|:---------------|:-----------|:-------------|:---------|
| sym 0%       | 60  | -0.31        | -0.57 to -0.08 | 0.0051         | 0.0510     | -0.61        | 7/32/21  |
| sym 10%      | 60  | +0.12        | -0.33 to +0.53 | 0.0678         | 0.4743     | +0.27        | 38/1/21  |
| sym 20%      | 59  | +0.40        | -0.44 to +1.23 | 0.1377         | 0.8259     | +0.23        | 34/3/22  |
| sym 30%      | 58  | +1.26        | +0.59 to +1.99 | 0.0023         | 0.0258     | +0.46        | 36/1/21  |
| sym 40%      | 57  | +0.77        | +0.15 to +1.44 | 0.0463         | 0.3707     | +0.34        | 28/12/17 |
| asym 10%     | 60  | +0.24        | -0.14 to +0.60 | 0.0324         | 0.2915     | +0.33        | 35/6/19  |
| asym 20%     | 60  | +0.34        | -0.22 to +0.89 | 0.1503         | 0.8259     | +0.23        | 33/7/20  |
| asym 30%     | 60  | +0.54        | -0.17 to +1.30 | 0.2332         | 0.8259     | +0.19        | 29/10/21 |
| asym 40%     | 60  | +2.20        | +1.08 to +3.52 | 0.0010         | 0.0122     | +0.61        | 28/22/10 |
| boundary 10% | 59  | -0.14        | -0.32 to -0.00 | 0.4537         | 0.9074     | -0.17        | 14/33/12 |
| boundary 20% | 55  | -0.12        | -0.39 to +0.13 | 0.7421         | 0.9074     | -0.07        | 14/25/16 |
| boundary 30% | 55  | -0.19        | -0.48 to +0.06 | 0.1650         | 0.8259     | -0.29        | 12/25/18 |

: **Table 4** LOO-EM (BIC) versus wide-grid ridge on the 60 confirmation datasets (columns as in Table 3)


![**Fig. 6** Per-dataset accuracy of LOO-EM against wide-grid ridge; colour encodes training-set size](/home/user/rp_c_14/figures/fig_scatter_em_vs_ridge.png){width=6.3in}

Against calibrated confident learning, the strongest cleaning baseline, LOO-EM is better on clean labels (+2.0 points, 95% CI +0.9 to +3.7, Holm-adjusted p < 0.001, 39 wins and 7 losses on the confirmation set) and at 10% asymmetric noise (+1.5 points, 95% CI +0.4 to +3.3, Holm-adjusted p = 0.017, 36 wins and 23 losses), and statistically indistinguishable from it at noise rates of 20% and above (−0.5 points, 95% CI −1.6 to +0.6, Holm-adjusted p = 1.000, 26 wins and 29 losses at 40% symmetric and +1.6 points, 95% CI +0.2 to +3.0, Holm-adjusted p = 0.353, 35 wins and 25 losses at 40% asymmetric noise). What separates the two is therefore not accuracy under heavy noise but the price paid when the noise is light or absent, and the computational cost (Sect. 6.9).

LOO-EM also yields cleaner training labels: after correction 87.3% of labels are correct at 20% symmetric noise (80% before) and 87.0% at 20% asymmetric noise. The cost on clean data is real: BIC accepts a noise model in 28% of clean runs, and LOO-EM is then slightly worse than leaving the labels alone (13 wins and 35 losses over all datasets). The estimated noise rate is conservative (mean 0.15 at a true rate of 0.20; Supplementary Fig. S1).


## 6.4 Ablation, and the failure of the first version

Table 5 and Supplementary Fig. S2 remove or alter one component at a time. *Leave-one-out scores are essential*: with in-sample fitted values the EM is indistinguishable from wide-grid ridge (63.7% versus 63.8% at 40% asymmetric noise) because BIC almost always prefers the noise-free model when the fit reproduces its targets. *The noise model is essential*: iterating on LOO probabilities without a transition matrix, i.e. LOO self-training, collapses to 73.0% on clean data. *Soft targets* matter little (hard targets: 65.2% versus 65.8% at 40% asymmetric noise). *The wide grid* matters under asymmetric and boundary noise (default grid: 64.2% and 75.8% versus 65.8% and 76.9%). *The symmetric model alone* matches the BIC choice except at 30–40% asymmetric noise, where the full matrix adds up to 1.8 points.

| Variant                               | S 0   | S 10   | S 20   | S 30   | S 40   | A 10   | A 20   | A 30   | A 40   | B 10   | B 20   | B 30   |
|:--------------------------------------|:------|:-------|:-------|:-------|:-------|:-------|:-------|:-------|:-------|:-------|:-------|:-------|
| LOO-EM (BIC, proposed)                | 86.8  | 84.1   | 80.2   | 74.8   | 66.8   | 84.7   | 81.9   | 74.1   | 65.8   | 83.1   | 80.0   | 76.9   |
| + instance-dependent model (rejected) | 86.8  | 84.1   | 80.2   | 74.7   | 66.8   | 84.7   | 81.8   | 73.8   | 65.6   | 83.1   | 80.3   | 77.3   |
| BIC: hard targets                     | 86.9  | 84.2   | 80.1   | 74.4   | 66.4   | 84.8   | 81.9   | 74.2   | 65.2   | 83.2   | 79.9   | 76.9   |
| BIC: in-sample scores                 | 87.2  | 83.9   | 79.6   | 73.9   | 65.8   | 84.5   | 81.3   | 73.2   | 63.7   | 83.2   | 80.0   | 76.8   |
| BIC: default grid                     | 86.7  | 84.1   | 80.5   | 75.0   | 67.0   | 84.4   | 81.7   | 73.5   | 64.2   | 82.9   | 79.7   | 75.8   |
| BIC: 5 EM iterations                  | 86.9  | 84.3   | 80.6   | 75.0   | 66.9   | 84.9   | 82.2   | 74.3   | 65.6   | 83.2   | 80.0   | 77.0   |
| BIC: penalty x0.5                     | 86.7  | 84.1   | 80.4   | 74.9   | 67.1   | 84.6   | 81.9   | 74.5   | 66.9   | 83.1   | 80.1   | 76.8   |
| BIC: penalty x2                       | 87.0  | 84.0   | 80.0   | 74.5   | 66.8   | 84.7   | 81.8   | 73.8   | 64.6   | 83.1   | 80.0   | 77.0   |
| Symmetric T only                      | 86.9  | 84.1   | 80.2   | 74.8   | 67.0   | 84.8   | 82.0   | 73.8   | 64.0   | 83.1   | 80.0   | 77.0   |
| LOO-EM v1 (full T)                    | 85.4  | 82.4   | 78.6   | 73.5   | 65.8   | 82.7   | 80.2   | 74.0   | 67.4   | 82.8   | 79.9   | 76.4   |
| v1 without gate                       | 84.5  | 81.7   | 78.3   | 74.2   | 66.1   | 82.5   | 79.9   | 73.8   | 67.2   | 82.4   | 79.6   | 76.1   |
| LOO self-training (no T)              | 73.0  | 67.6   | 61.6   | 55.0   | 48.4   | 67.8   | 62.4   | 56.4   | 48.4   | 73.3   | 72.4   | 68.3   |
| LOO relabel (1 round)                 | 82.6  | 80.4   | 77.0   | 71.3   | 64.2   | 80.7   | 78.4   | 71.2   | 62.3   | 82.1   | 80.7   | 78.3   |
| LOO relabel (iterated)                | 79.6  | 77.3   | 74.2   | 69.6   | 62.7   | 77.5   | 75.5   | 69.8   | 60.7   | 79.2   | 77.9   | 75.5   |
| LOO remove                            | 84.1  | 82.2   | 79.1   | 73.5   | 66.2   | 82.5   | 80.7   | 73.9   | 65.5   | 82.8   | 80.7   | 78.1   |
| Ridge (wide grid)                     | 87.2  | 83.9   | 79.6   | 73.6   | 65.8   | 84.5   | 81.3   | 73.5   | 63.8   | 83.2   | 80.1   | 76.9   |

: **Table 5** Ablation and variants: mean test accuracy (%) over all datasets. S, A and B denote symmetric, asymmetric and boundary noise at the given rate (%)



The first version (v1) used the full matrix whenever a one-parameter likelihood gate accepted it. On the 12 development datasets, none of which has more than eight classes, it looked uniformly beneficial. On all 108 datasets it loses 1.8 points on clean labels (12 wins and 59 losses) and 1.5 points at 10% symmetric noise, and gains 3.6 points at 40% asymmetric noise, more than the BIC variant (2.0). The loss is concentrated where the matrix is over-parameterised: for the datasets with ten or more classes v1 scores 79.2% on clean data against 83.1% for ridge, yet at 40% asymmetric noise it is the only method that helps there (58.8% versus 54.0% for the BIC variant). This is the trade-off of Sect. 4.4 in numbers: under 40% asymmetric noise BIC accepts a noise model in only 36% of runs, because data in which 40% of each class is consistently relabelled as its neighbour are equally well explained as two overlapping classes. Halving the BIC penalty moves the method towards v1 (66.9% at 40% asymmetric noise, 86.7% on clean labels), doubling it moves it towards ridge (64.6% and 87.0%).

**Does the global standardisation matter?** The closed form keeps the column scaling computed on all training series, so its LOO values differ slightly from those of a pipeline that re-standardises without the left-out series. For the 41 datasets with at most 100 training series we computed the exact operator by refitting the scaling and the ridge model $n$ times for every penalty. On clean labels the median correlation between closed-form and exact LOO decision values was 0.9997 (minimum 0.986) and the predicted class agreed for 99.4% of training series. The selected penalty, however, was identical in only 49% of datasets and within one grid step in 56%: on clean labels the leave-one-out criterion is flat over a wide range of penalties, so the choice within that range is unstable while the predictions it implies are not. Running LOO-EM with the exact operator changed mean test accuracy by 0.0 points on clean labels, −1.6 points under 30% symmetric noise, −0.3 points under 40% asymmetric noise; the largest difference (30% symmetric noise) has Wilcoxon p = 0.10, and none is significant at the 5% level, so the approximation does not change the conclusions. Refitting the random-kernel transform itself without each series, the remaining approximation, is shared by every LOO-based method including the standard ridge pipeline.

**An instance-dependent noise model (rejected).** Boundary noise, where the flipped labels are those of uncertain series, is the case LOO-EM does not help. We therefore tested a two-parameter extension in which the flip probability rises with the LOO uncertainty of each series and flips go preferentially to classes the classifier finds plausible (Xia et al. 2020 discuss this class of noise), added as a fourth candidate to the BIC choice. Its adoption rule was fixed before it was run (protocol, stage 3): on the design set it gained 0.7 points at 30% boundary noise but lost 0.6 and 0.8 points at 20% and 40% asymmetric noise, so it was rejected. On the confirmation datasets the picture was the same (+0.4 points, 95% CI −0.1 to +1.0, Holm-adjusted p = 1.000, 12 wins and 12 losses at 30% boundary noise), and on the ECG data it behaved exactly like LOO-EM (Sect. 6.6). Because the synthetic boundary noise flips low-margin series to the runner-up class, a mechanism close to this model, even a gain there would have been weak evidence.

## 6.5 Robustness: training-set size, number of classes, noise model

Table 6 stratifies the main comparison. LOO-EM's gain over wide-grid ridge at 40% asymmetric noise grows with the training-set size: +0.8 points below 100 series, +2.1 between 100 and 500, and +3.8 for 500 or more. With fewer than 100 training series no method is reliably better than another under symmetric noise, and all methods lose more than 20 points at 30% noise. By number of classes, LOO-EM helps most for 3–9 classes (+4.4 points at 40% asymmetric noise), is neutral for two classes, where pair-flip noise is a single flip probability that the symmetric model already represents, and does not help with ten or more classes at that noise level.

| Stratum   | Noise        | Datasets   | Ridge (default)   | Ridge (wide)   | LOO-EM (BIC)   | LOO-EM v1   | Confident learning (cal.)   |
|:----------|:-------------|:-----------|:------------------|:---------------|:---------------|:------------|:----------------------------|
| n<100     | sym 0%       | 36         | 92.4              | 92.4           | 92.2           | 91.0        | 90.6                        |
| n<100     | sym 30%      | 36         | 70.5              | 68.5           | 70.4           | 70.0        | 71.7                        |
| n<100     | asym 40%     | 36         | 61.3              | 64.8           | 65.6           | 65.7        | 63.9                        |
| n<100     | boundary 20% | 35         | 79.5              | 80.3           | 80.2           | 80.4        | 82.3                        |
| 100≤n<500 | sym 0%       | 51         | 81.9              | 82.1           | 81.5           | 79.6        | 79.1                        |
| 100≤n<500 | sym 30%      | 49         | 69.3              | 72.3           | 73.2           | 71.5        | 73.0                        |
| 100≤n<500 | asym 40%     | 51         | 55.6              | 58.8           | 60.9           | 63.3        | 59.6                        |
| 100≤n<500 | boundary 20% | 45         | 76.5              | 77.4           | 77.4           | 77.3        | 77.2                        |
| n≥500     | sym 0%       | 21         | 90.7              | 90.7           | 90.6           | 89.7        | 90.2                        |
| n≥500     | sym 30%      | 21         | 81.8              | 85.4           | 86.3           | 84.2        | 86.0                        |
| n≥500     | asym 40%     | 21         | 66.2              | 74.5           | 78.2           | 80.4        | 77.2                        |
| n≥500     | boundary 20% | 21         | 84.7              | 85.4           | 85.4           | 84.7        | 85.3                        |
| C=2       | sym 0%       | 38         | 91.3              | 91.3           | 91.3           | 91.1        | 90.6                        |
| C=2       | sym 30%      | 38         | 72.1              | 73.9           | 74.4           | 74.4        | 76.4                        |
| C=2       | asym 40%     | 38         | 62.6              | 67.7           | 67.6           | 67.5        | 68.0                        |
| C=2       | boundary 20% | 38         | 83.1              | 84.3           | 84.4           | 84.4        | 85.3                        |
| 3≤C<10    | sym 0%       | 49         | 85.5              | 85.7           | 85.2           | 83.5        | 83.5                        |
| 3≤C<10    | sym 30%      | 49         | 72.3              | 73.7           | 75.2           | 74.0        | 75.1                        |
| 3≤C<10    | asym 40%     | 49         | 59.3              | 65.2           | 69.6           | 71.1        | 66.0                        |
| 3≤C<10    | boundary 20% | 48         | 76.8              | 77.6           | 77.5           | 77.5        | 78.1                        |
| C≥10      | sym 0%       | 21         | 83.3              | 83.1           | 82.7           | 79.2        | 78.6                        |
| C≥10      | sym 30%      | 19         | 72.1              | 72.8           | 74.8           | 70.6        | 72.4                        |
| C≥10      | asym 40%     | 21         | 54.9              | 53.8           | 54.0           | 58.8        | 54.5                        |
| C≥10      | boundary 20% | 15         | 77.1              | 77.3           | 77.3           | 76.1        | 77.2                        |

: **Table 6** Mean test accuracy (%) by training-set size and by number of classes


Boundary noise is the hardest case for every label-based remedy. The mislabelled series are those the clean model was least sure about, relabelled to the class it considered second most likely; the widened grid recovers 2.2 points at the 30% rate, LOO-EM adds nothing (0.0 points, 95% CI −0.3 to +0.2, Holm-adjusted p = 1.000, 28 wins and 32 losses), and simple LOO relabelling is the best method (78.3%).

**Class-wise behaviour and calibration.** A further pass over all datasets and seeds recorded the recall of the smallest training class, the worst per-class recall and the calibration of probabilities obtained from each final model's own LOO scores (Fig. 7). It shows the cost of the wide grid more starkly than balanced accuracy does. At 40% symmetric noise the worst per-class recall is 41.0% with the default grid and 29.8% with the wide grid (−11.2 points, Holm-adjusted p < 0.001, 19 wins and 83 losses), and the recall of the smallest class falls from 57.7% to 52.9%; at 40% asymmetric noise the worst-class recall is 40.5% against 34.2%. The extra shrinkage that buys accuracy is paid for by the hardest class. LOO-EM does not restore it (30.0% at 40% symmetric noise), whereas calibrated confident learning partly does under boundary noise (49.3% against 46.7% at 30%). Calibration is where LOO-EM differs most from plain ridge: the expected calibration error at 20% symmetric noise is 18.8% for wide-grid ridge and 11.7% for LOO-EM (−7.2 points, Holm-adjusted p < 0.001, lower on 84 and higher on 20 datasets) and at 40% 23.4% against 16.3%, while on clean labels LOO-EM is slightly worse calibrated (6.5% against 4.8%). A temperature fitted to corrupted targets reproduces their uncertainty; soft targets from the EM remove part of it.

![**Fig. 7** Class-wise behaviour and calibration against the noise rate (mean over datasets): recall of the smallest training class, worst per-class recall and expected calibration error of temperature-scaled probabilities; solid lines symmetric, dashed lines asymmetric noise](/home/user/rp_c_14/figures/fig_classwise.png){width=6.3in}

## 6.6 RQ5: real annotation errors in 8,528 ECG recordings

Table 7 and Fig. 8 report the ECG experiment. Three results stand out.

*Real annotation noise cost the regularised classifier little.* Wide-grid ridge trained on the original labels agrees with the expert-revised labels on 72.0% of recordings; trained on the expert labels it reaches 72.6% (+0.6 accuracy points, 95% CI +0.3 to +0.9; F1 +0.006). Default-grid ridge trained on the original labels already reaches 72.6% (challenge F1 0.651). With 4.8% of labels wrong, most of them recordings later judged too noisy to classify, the classifier's own error of about 27% dominates. All paired tests are listed in Table 8.

*No relabelling or removal method improved agreement with the expert labels, and most reduced it, LOO-EM included.* Relative to wide-grid ridge, LOO-EM changed accuracy by −1.5 accuracy points, 95% CI −2.1 to −0.9; F1 −0.028; this is the first pre-specified primary comparison, Holm-adjusted p < 0.001. BIC chose the full transition matrix in all ten folds, and the EM altered 15.7% of the training labels although only 4.8% were documented as wrong; agreement of the training labels with the expert revision fell from 95.2% to 81.2%. The corrections were not random: recall of atrial fibrillation rose from 41.6% to 60.4%, while recall of the heterogeneous "other rhythm" class fell from 43.4% to 22.8%. The EM had re-explained the classifier's systematic confusion between normal and other rhythms as label noise, the identifiability failure anticipated in Sect. 4.4, and with about 6,822 training recordings per fold the BIC penalty of $\tfrac{k}{2}\log n$ was far too small to stop it. One-round LOO relabelling (−3.2 accuracy points, 95% CI −3.7 to −2.6; F1 −0.087), LOO removal (−1.5 accuracy points, 95% CI −1.9 to −1.0; F1 −0.039) and edited nearest neighbours (−2.4 accuracy points, 95% CI −3.0 to −1.9; F1 −0.049) were also worse; calibrated confident learning was indistinguishable from doing nothing (−0.3 accuracy points, 95% CI −0.6 to +0.1; F1 −0.002). The only correction that helped was backward loss correction, which had been the weakest method on synthetic noise: +0.4 accuracy points, 95% CI +0.1 to +0.6; F1 +0.005. It raised recall of the noisy class from 0.0% to 12.0%, consistent with the dominant real error, noisy recordings labelled as normal or other rhythm, being class-conditional and towards a single under-labelled class.

*Detection of the documented errors was modest for every score.* The AUROC of the suspicion scores against the 412 revisions was 0.69 for the closed-form LOO margin, 0.69 for 5-fold cross-validated scores, 0.69 for the LOO-EM posterior and 0.64 for nearest-neighbour disagreement. Among as many top-ranked recordings as there were revisions, 8.2% (LOO margin) were revised ones, compared with a base rate of 4.8%: the ranking concentrates true errors by a factor of about 1.7 but is far from a reliable error detector. The closed-form score matched the 5-fold one at a fraction of the cost.

*The penalty grid did not help either.* The second primary comparison went against the wide grid: −0.6 accuracy points, 95% CI −1.0 to −0.2; F1 −0.024, Holm-adjusted p = 0.024. The default grid was saturated at 10^3^ in every fold, and the leave-one-out squared error chose a larger penalty on the wide grid even when the labels were the expert ones, so the loss is not caused by noise: the squared-error criterion shrinks towards the majority class (normal rhythm, 60% of recordings) and gives up recall of atrial fibrillation (49.3% with the default grid, 41.6% with the wide one). This is the metric dependence of Sect. 6.1 seen on real data; Sect. 6.7 examines the selection criterion.

| Method                                        | Acc.   | F1 (NAO)   | Bal. acc.   | Rec. N   | Rec. A   | Rec. O   | Rec. ~   | Acc. vs original   |
|:----------------------------------------------|:-------|:-----------|:------------|:---------|:---------|:---------|:---------|:-------------------|
| Ridge (default grid)                          | 72.6   | 0.651      | 47.0        | 93       | 49       | 46       | 0        | 74.3               |
| Ridge (wide grid)                             | 72.0   | 0.627      | 44.8        | 94       | 42       | 43       | 0        | 73.8               |
| LOO-EM (BIC, proposed)                        | 70.5   | 0.599      | 47.6        | 98       | 60       | 23       | 9        | 71.6               |
| LOO-EM v1 (full T)                            | 70.5   | 0.599      | 47.6        | 98       | 60       | 23       | 9        | 71.6               |
| LOO-EM + instance model (rejected)            | 70.5   | 0.599      | 47.6        | 98       | 60       | 23       | 9        | 71.6               |
| Confident learning (calibrated)               | 71.7   | 0.625      | 44.8        | 93       | 41       | 45       | 0        | 73.6               |
| Confident learning (uncalibrated)             | 71.8   | 0.631      | 45.3        | 92       | 43       | 46       | 0        | 73.8               |
| 5-fold relabel                                | 71.2   | 0.609      | 43.4        | 95       | 38       | 41       | 0        | 72.9               |
| ENN                                           | 69.5   | 0.578      | 41.7        | 97       | 41       | 28       | 0        | 71.1               |
| LOO relabel (1 round)                         | 68.8   | 0.540      | 38.6        | 96       | 25       | 33       | 0        | 70.5               |
| LOO remove                                    | 70.5   | 0.588      | 41.9        | 97       | 36       | 35       | 0        | 72.2               |
| Backward-corrected ridge                      | 72.3   | 0.632      | 48.1        | 94       | 44       | 42       | 12       | 73.7               |
| Ridge (wide), trained on expert labels        | 72.6   | 0.633      | 49.1        | 95       | 43       | 42       | 16       | 73.5               |
| Ridge (wide, LOO-error criterion) †           | 72.6   | 0.651      | 47.0        | 93       | 49       | 46       | 0        | 74.3               |
| Ridge (wide, LOO balanced-error criterion) †  | 70.5   | 0.645      | 47.6        | 87       | 54       | 49       | 1        | 72.2               |
| Ridge (wide, balanced-error), expert labels † | 72.0   | 0.659      | 53.5        | 89       | 56       | 48       | 21       | 72.7               |

: **Table 7** ECG data (PhysioNet/CinC 2017): methods trained on the original labels, evaluated against the expert-revised labels on pooled out-of-fold predictions (5-fold cross-validation, two repetitions). Accuracy, the challenge F1 (mean of N, A, O), balanced accuracy and per-class recall in %; the last column scores the same predictions against the original labels. Rows trained on expert labels are references, not competitors. † exploratory penalty criteria (Sect. 6.7)


| Comparison                                                                                | Δ accuracy (points) [95% CI]   | Δ F1 [95% CI]           | McNemar p (rep. 0, 1)   | Holm p (primary)   |
|:------------------------------------------------------------------------------------------|:-------------------------------|:------------------------|:------------------------|:-------------------|
| LOO-EM vs ridge (wide)                                                                    | −1.50 [−2.12, −0.87]           | −0.028 [−0.039, −0.017] | <0.001, <0.001          | <0.001             |
| ridge (wide) vs ridge (default)                                                           | −0.59 [−0.96, −0.23]           | −0.024 [−0.031, −0.018] | 0.007, 0.024            | 0.024              |
| LOO-EM vs Confident learning (calibrated)                                                 | −1.24 [−1.90, −0.60]           | −0.027 [−0.037, −0.015] | <0.001, 0.004           |                    |
| LOO-EM vs ridge (default)                                                                 | −2.09 [−2.74, −1.45]           | −0.053 [−0.063, −0.043] | <0.001, <0.001          |                    |
| Confident learning (calibrated) vs ridge (wide)                                           | −0.25 [−0.61, +0.11]           | −0.002 [−0.010, +0.006] | 0.345, 0.270            |                    |
| LOO relabel (1 round) vs ridge (wide)                                                     | −3.15 [−3.67, −2.63]           | −0.087 [−0.099, −0.077] | <0.001, <0.001          |                    |
| LOO remove vs ridge (wide)                                                                | −1.47 [−1.92, −1.01]           | −0.039 [−0.049, −0.031] | <0.001, <0.001          |                    |
| ENN vs ridge (wide)                                                                       | −2.44 [−3.00, −1.90]           | −0.049 [−0.060, −0.039] | <0.001, <0.001          |                    |
| Backward-corrected ridge vs ridge (wide)                                                  | +0.39 [+0.14, +0.64]           | +0.005 [+0.001, +0.010] | 0.013, 0.021            |                    |
| LOO-EM v1 (full T) vs ridge (wide)                                                        | −1.50 [−2.12, −0.87]           | −0.028 [−0.039, −0.017] | <0.001, <0.001          |                    |
| LOO-EM + instance model (rejected) vs ridge (wide)                                        | −1.50 [−2.12, −0.87]           | −0.028 [−0.039, −0.017] | <0.001, <0.001          |                    |
| Ridge (wide), trained on expert labels vs ridge (wide)                                    | +0.60 [+0.28, +0.92]           | +0.006 [−0.001, +0.012] | 0.003, 0.002            |                    |
| Ridge (wide, LOO balanced-error criterion) vs ridge (wide)                                | −1.42 [−2.03, −0.81]           | +0.018 [+0.008, +0.027] | <0.001, 0.003           |                    |
| Ridge (wide, LOO-error criterion) vs ridge (wide)                                         | +0.59 [+0.23, +0.96]           | +0.024 [+0.018, +0.031] | 0.007, 0.024            |                    |
| Ridge (wide, LOO balanced-error criterion) vs ridge (default)                             | −2.01 [−2.49, −1.53]           | −0.006 [−0.013, +0.002] | <0.001, <0.001          |                    |
| Ridge (wide), trained on expert labels vs ridge (default)                                 | +0.01 [−0.42, +0.45]           | −0.019 [−0.027, −0.011] | 0.899, 0.825            |                    |
| Ridge (wide, balanced-error), expert labels vs Ridge (wide, LOO balanced-error criterion) | +1.47 [+1.05, +1.91]           | +0.014 [+0.007, +0.021] | <0.001, <0.001          |                    |

: **Table 8** ECG data: paired comparisons against the expert-revised labels. Differences of accuracy and challenge F1 with paired-bootstrap 95% confidence intervals over recordings; McNemar's exact test on per-recording correctness, separately for each repetition; Holm correction over the two pre-specified primary comparisons only. Comparisons involving the exploratory criteria or expert-label references are descriptive


![**Fig. 8** PhysioNet/CinC 2017 ECG data, original labels for training and expert-revised labels for evaluation. Left: ROC curves of four suspicion scores for finding the documented label revisions on the training folds (repetition 0). Right: confusion matrices against the expert-revised labels (row-normalised, %) for wide-grid ridge and LOO-EM trained on the original labels and for wide-grid ridge trained on the expert labels](/home/user/rp_c_14/figures/fig_cinc2017.png){width=6.3in}

## 6.7 Choosing the penalty by the target metric (exploratory)

The ECG result and the metric dependence of Sect. 6.1 have a common cause: the leave-one-out squared error rewards shrinking the decision function towards the class-frequency intercept, which costs minority-class recall. Because this explanation was formed after seeing the ECG data, the following comparison is exploratory (stage 4 of the protocol); the UCR runs, on which these variants had not been run before, are its test. We kept the wide grid and changed only the criterion that selects the penalty from it: the leave-one-out classification error or the leave-one-out balanced error (mean per-class error), both read off the same closed-form LOO decision values, with ties broken by the squared error.

Over the 108 UCR datasets, the balanced-error criterion improves on the default pipeline under every noise model: +2.8 points, 95% CI +1.7 to +3.9, Holm-adjusted p < 0.001, 71 wins and 31 losses at 40% symmetric noise and +2.9 points, 95% CI +1.7 to +4.2, Holm-adjusted p < 0.001, 76 wins and 30 losses at 40% asymmetric noise. Relative to the squared-error criterion on the same grid it trades accuracy for balance: at 40% symmetric noise balanced accuracy is 64.7% against 62.4% and macro-F1 61.9% against 57.9%, removing the minority-class penalty of the wide grid, but accuracy at 40% asymmetric noise is 62.5% against 63.8% and at 30% boundary noise 76.2% against 76.9%. The plain error criterion lies in between. On clean labels all three criteria agree to within 0.1 points. Table 9 gives the full comparison.

| Penalty selection             | Metric            | sym 0%   | sym 20%   | sym 40%   | asym 20%   | asym 40%   | boundary 30%   |
|:------------------------------|:------------------|:---------|:----------|:----------|:-----------|:-----------|:---------------|
| Default grid, squared error   | Accuracy          | 87.1     | 79.1      | 64.1      | 79.6       | 59.6       | 74.7           |
| Default grid, squared error   | Balanced accuracy | 85.1     | 77.1      | 62.5      | 77.8       | 58.2       | 72.1           |
| Wide grid, squared error      | Accuracy          | 87.2     | 79.6      | 65.8      | 81.3       | 63.8       | 76.9           |
| Wide grid, squared error      | Balanced accuracy | 85.1     | 77.0      | 62.4      | 78.7       | 60.8       | 73.8           |
| Wide grid, LOO error          | Accuracy          | 87.1     | 80.0      | 66.2      | 81.3       | 63.1       | 76.1           |
| Wide grid, LOO error          | Balanced accuracy | 85.0     | 77.5      | 63.2      | 78.9       | 60.6       | 72.8           |
| Wide grid, LOO balanced error | Accuracy          | 87.1     | 80.0      | 66.9      | 81.1       | 62.5       | 76.2           |
| Wide grid, LOO balanced error | Balanced accuracy | 85.1     | 77.9      | 64.7      | 79.0       | 60.6       | 73.4           |
| LOO-EM (BIC)                  | Accuracy          | 86.8     | 80.2      | 66.8      | 81.9       | 65.8       | 76.9           |
| LOO-EM (BIC)                  | Balanced accuracy | 84.8     | 77.6      | 63.6      | 79.4       | 62.8       | 73.9           |

: **Table 9** Exploratory: penalty-selection criterion on the wide grid, mean test accuracy and balanced accuracy (%) over 108 UCR datasets


On the ECG data the balanced-error criterion reached 70.5% accuracy and challenge F1 0.645, against 72.0% and 0.627 for the squared error on the same grid (−1.4 accuracy points, 95% CI −2.0 to −0.8; F1 +0.018) and 72.6% and 0.651 for the default grid. It thus traded accuracy for the class-balanced challenge metric, as on the UCR data, but did not improve on the default grid on either metric. The plain error criterion, by contrast, reached 72.6% and 0.651, recovering what the squared error had lost on the same grid (+0.6 accuracy points, 95% CI +0.2 to +1.0; F1 +0.024). Since the ECG data suggested the analysis, they are not independent evidence. The conclusion we draw is limited: the criterion that selects the penalty should match the metric of interest, and different criteria can be compared at no cost because all are read off the same closed-form predictions; the squared-error default is not neutral with respect to class balance.

## 6.8 RQ4: generalisation

Table 10 summarises three checks of external validity for the kernel-transform pipeline.

*Hydra features* (29 datasets, seeds 0–2). The datasets are the fixed random subset of the sensitivity study, without StarLightCurves, whose Hydra transform exceeded the available memory. Hydra counts are square-rooted and standardised here rather than passed through the sparse scaler of the original pipeline. The wide grid changes accuracy relative to the default grid by −0.8 points, Holm-adjusted p = 1.000, 6 wins and 4 losses on clean labels, +2.5 points, Holm-adjusted p = 0.306, 18 wins and 10 losses at 40% symmetric and +4.7 points, Holm-adjusted p = 0.026, 23 wins and 6 losses at 40% asymmetric noise. LOO-EM relative to wide-grid ridge: 0.0 points, Holm-adjusted p = 1.000, 4 wins and 10 losses on clean labels, +1.4 points, Holm-adjusted p = 0.165, 15 wins and 5 losses at 40% symmetric and +0.5 points, Holm-adjusted p = 1.000, 9 wins and 4 losses at 40% asymmetric noise.

*Multivariate series* (18 UEA datasets, seeds 0–2). Wide versus default grid: +2.4 points, Holm-adjusted p = 0.907, 13 wins and 5 losses at 40% symmetric and +3.9 points, Holm-adjusted p = 0.182, 13 wins and 5 losses at 40% asymmetric noise. LOO-EM versus wide-grid ridge: −0.9 points, Holm-adjusted p = 0.252, 1 win and 6 losses on clean labels and +1.4 points, Holm-adjusted p = 0.261, 8 wins and 4 losses at 40% asymmetric noise. With this few datasets the tests have little power and the table should be read descriptively.

*Larger training sets* (Crop, FordA, FordB; 3,601–7,200 training series, seeds 0–2; no significance tests with 3 datasets). At 40% asymmetric noise the default grid gives 60.1%, the wide grid 78.2%, LOO-EM 77.8% and v1 76.7%; at 40% symmetric noise 64.4%, 79.9%, 78.7% and 77.1%; on clean labels 86.8%, 86.6%, 86.6% and 86.6%.

| Experiment                                 | Method                            | sym 0%   | sym 20%   | sym 40%   | asym 20%   | asym 40%   | boundary 20%   |
|:-------------------------------------------|:----------------------------------|:---------|:----------|:----------|:-----------|:-----------|:---------------|
| Hydra (29 UCR datasets)                    | Ridge (default grid)              | 84.6     | 77.4      | 62.2      | 77.7       | 57.8       | 80.5           |
| Hydra (29 UCR datasets)                    | Ridge (wide grid)                 | 83.8     | 77.7      | 64.7      | 78.3       | 62.5       | 81.1           |
| Hydra (29 UCR datasets)                    | LOO-EM (BIC)                      | 83.9     | 77.4      | 66.2      | 78.8       | 63.0       | 81.2           |
| Hydra (29 UCR datasets)                    | LOO-EM v1                         | 82.6     | 74.7      | 65.4      | 77.0       | 65.3       | 80.5           |
| Hydra (29 UCR datasets)                    | Confident learning (uncalibrated) | 81.5     | 77.1      | 67.5      | 78.2       | 64.3       | 81.6           |
| MiniRocket, multivariate (18 UEA datasets) | Ridge (default grid)              | 74.3     | 69.8      | 57.8      | 69.5       | 54.1       | 70.5           |
| MiniRocket, multivariate (18 UEA datasets) | Ridge (wide grid)                 | 74.2     | 69.6      | 60.2      | 69.9       | 58.0       | 70.6           |
| MiniRocket, multivariate (18 UEA datasets) | LOO-EM (BIC)                      | 73.3     | 70.4      | 62.0      | 70.8       | 59.4       | 70.4           |
| MiniRocket, multivariate (18 UEA datasets) | LOO-EM v1                         | 72.9     | 68.1      | 61.4      | 69.1       | 62.1       | 69.7           |
| MiniRocket, multivariate (18 UEA datasets) | Confident learning (uncalibrated) | 71.1     | 68.1      | 59.6      | 68.6       | 59.4       | 70.9           |
| MiniRocket, largest UCR datasets (3)       | Ridge (default grid)              | 86.8     | 78.6      | 64.4      | 77.9       | 60.1       | 83.1           |
| MiniRocket, largest UCR datasets (3)       | Ridge (wide grid)                 | 86.6     | 84.9      | 79.9      | 85.0       | 78.2       | 84.4           |
| MiniRocket, largest UCR datasets (3)       | LOO-EM (BIC)                      | 86.6     | 85.6      | 78.7      | 85.4       | 77.8       | 84.4           |
| MiniRocket, largest UCR datasets (3)       | LOO-EM v1                         | 86.6     | 83.3      | 77.1      | 83.8       | 76.7       | 84.4           |
| MiniRocket, largest UCR datasets (3)       | ENN                               | 85.3     | 83.9      | 76.6      | 83.9       | 75.7       | 83.7           |

: **Table 10** Generalisation: mean test accuracy (%) with other random-kernel transforms, on multivariate UEA datasets and on the largest UCR datasets (seeds 0–2)


*Other classifier families* (Table 11; 108 datasets, seeds 0–2). QUANT, whose extremely randomised trees average over many weak learners, loses 19.7 points between clean labels and 40% symmetric noise (86.1% to 66.5%), compared with 22.8 points for default-grid MiniRocket and 21.6 for the wide grid. At 40% asymmetric noise QUANT reaches 61.8%, wide-grid ridge 63.8% and LOO-EM 65.9% (LOO-EM versus QUANT: +4.1 points, Holm-adjusted p < 0.001, 74 wins and 34 losses; at 40% symmetric noise −0.2 points, Holm-adjusted p = 1.000, 53 wins and 52 losses). The 1-nearest-neighbour classifier is the most fragile method tested (71.9% clean, 48.7% at 40% symmetric noise).

*Deep baselines* (Supplementary Table S1; 6 small datasets, seeds 0–2, CPU training). At 30% symmetric noise a fully convolutional network reaches 73.2% with cross-entropy, 68.3% with the generalised cross-entropy and 82.0% with Co-teaching given the true noise rate; wide-grid ridge reaches 78.1% and LOO-EM 78.1%. At 30% asymmetric noise the figures are 72.1%, 72.7%, 78.3%, 80.9% and 80.1%; on clean labels 91.1%, 93.6%, 92.1%, 94.4% and 94.4%. One network takes 17 s on average on one CPU core (Co-teaching 35 s). On these datasets Co-teaching, which is told the true noise rate, is the most accurate method at 30% symmetric noise, and the ridge variants are the most accurate on clean labels and under asymmetric noise. The comparison covers few, small datasets and simple networks and supports no general claim about deep noisy-label methods.

| Experiment       | Method               | sym 0%   | sym 20%   | sym 40%   | asym 20%   | asym 40%   | boundary 20%   |
|:-----------------|:---------------------|:---------|:----------|:----------|:-----------|:-----------|:---------------|
| 108 UCR datasets | Ridge (default grid) | 86.8     | 78.9      | 64.0      | 79.4       | 59.5       | 79.0           |
| 108 UCR datasets | Ridge (wide grid)    | 86.9     | 79.5      | 65.3      | 81.0       | 63.8       | 79.8           |
| 108 UCR datasets | LOO-EM (BIC)         | 86.5     | 80.0      | 66.3      | 81.5       | 65.9       | 80.0           |
| 108 UCR datasets | QUANT                | 86.1     | 80.0      | 66.5      | 80.5       | 61.8       | 78.6           |
| 108 UCR datasets | 1-NN Euclidean       | 71.9     | 60.8      | 48.7      | 60.2       | 48.2       | 67.2           |

: **Table 11** Other classifier families trained on the same noisy labels (seeds 0–2): mean test accuracy (%)


*A time-series-specific noisy-label method.* On the seven equal-length UCR datasets of the CTW authors' own experiment list (Table 12), CTW reaches 85.3% on clean labels, 61.3% at 40% symmetric and 64.0% at 40% asymmetric noise, against 96.2%, 68.9% and 71.7% for wide-grid ridge and 95.9%, 71.7% and 73.8% for LOO-EM (CTW versus LOO-EM at 40% asymmetric noise: −9.7 points, 1 win and 6 losses over 7 datasets; at 40% symmetric noise −10.4 points, 1 win and 6 losses over 7 datasets). Averaged over the six conditions, CTW is 10.6 points less accurate than LOO-EM and 9.5 points less accurate than wide-grid ridge. One CTW run took a median of 163 s on one CPU core, against well under a second for every ridge-based method on these datasets, a difference of more than two orders of magnitude. Seven datasets are too few for a significance test to carry much weight, and CTW was run with its published hyper-parameters rather than tuned; the comparison therefore locates the methods relative to each other on the CTW authors' own benchmark rather than establishing a general ranking. To check the re-implementation we also ran it under the CTW paper's own protocol (stratified 5-fold split of the pooled data, 30% symmetric noise), where it reached 0.952 on CBF (fold 0; published 0.958) and 0.806 on ArrowHead (mean of 5 folds, SD 0.082; published 0.851), within the published run-to-run variation; the lower accuracies on our splits therefore reflect their much smaller training sets (for example 30 series for CBF and 36 series for ArrowHead) rather than an implementation error.

| Noise        | Ridge (default)   | Ridge (wide)   | LOO-EM (BIC)   | Confident learning (cal.)   | CTW   |
|:-------------|:------------------|:---------------|:---------------|:----------------------------|:------|
| sym 0%       | 96.2              | 96.2           | 95.9           | 95.2                        | 85.3  |
| sym 20%      | 88.8              | 88.6           | 90.2           | 89.0                        | 76.6  |
| sym 40%      | 71.2              | 68.9           | 71.7           | 73.4                        | 61.3  |
| asym 20%     | 88.1              | 89.4           | 89.7           | 90.5                        | 79.7  |
| asym 40%     | 65.9              | 71.7           | 73.8           | 70.1                        | 64.0  |
| boundary 30% | 73.5              | 76.8           | 76.8           | 79.0                        | 67.5  |

: **Table 12** Comparison with CTW (re-implemented) on the UCR datasets of the CTW paper's experiment list (ArrowHead, CBF, FaceFour, OSULeaf, Plane, Symbols, Trace; MelbournePedestrian, which has unequal-length series, is excluded); mean test accuracy (%), seeds 0, 1, 2, identical splits and noise for all methods


## 6.9 Computational cost

Table 13 and Supplementary Fig. S3 report the time each method adds to the shared work, which is the MiniRocket transform (median 0.52 s per dataset) and one eigendecomposition (median 0.028 s). Selecting the penalty on the wide instead of the default grid costs 1 ms at the median. LOO-EM takes a median of 0.11 s (mean 0.28 s), 2.6 times the eigendecomposition; 5-fold confident learning takes 0.39 s (mean 1.04 s), 8.8 times. LOO-EM is therefore not free, as a single LOO relabelling round is (4 ms), but it is about 4 times cheaper on average than the cross-validated alternative and never refits a model. Memory is dominated by the $n\times n$ eigenvector matrix and its element-wise square. All ridge variants deploy the same linear classifier, so model size and inference cost are identical; energy use was not measured. On the largest dataset processed (Crop, 7,200 training series, two BLAS threads) the eigendecomposition takes 47 s and LOO-EM 27 s per noise condition.

| Method                          | Median time (s)   | Mean time (s)   | Median ratio to eigendecomposition   |
|:--------------------------------|:------------------|:----------------|:-------------------------------------|
| Ridge (default grid)            | 0.001             | 0.010           | 0.05                                 |
| Ridge (wide grid)               | 0.002             | 0.019           | 0.09                                 |
| LOO-EM (BIC, proposed)          | 0.108             | 0.278           | 2.56                                 |
| LOO-EM v1 (full T)              | 0.097             | 0.439           | 3.01                                 |
| Confident learning (calibrated) | 0.388             | 1.043           | 8.84                                 |
| 5-fold relabel                  | 0.145             | 0.757           | 4.87                                 |
| ENN                             | 0.116             | 0.442           | 2.20                                 |
| LOO relabel (1 round)           | 0.004             | 0.035           | 0.18                                 |
| LOO remove                      | 0.101             | 0.458           | 2.07                                 |

: **Table 13** Wall-clock cost per run of each method beyond the shared feature transform and eigendecomposition (single CPU thread, all datasets and conditions)



**Scaling with the training-set size.** Fig. 9 measures decomposition time, LOO-EM time and peak memory on ElectricDevices for training sets of 500 to 14,000 series and two feature dimensions, each configuration in a fresh process. With 1,932 features and n = 14,000 the primal route took 24 s to decompose, LOO-EM 20 s, and peak memory was 1.4 GB. With 9,996 features and n = 14,000 the primal route took 346 s to decompose, LOO-EM 74 s, and peak memory was 6.1 GB. Below the feature dimension the cost grows cubically with n (dual route); above it, the primal route makes the cost linear in n for a fixed feature dimension, so that reducing the number of kernels is the lever for very large training sets.

![**Fig. 9** Scalability on ElectricDevices with 30% asymmetric noise: decomposition and LOO-EM time, peak resident memory and test accuracy against training-set size for two feature dimensions; the primal route is used when n exceeds the number of features](/home/user/rp_c_14/figures/fig_scaling.png){width=6.3in}

**Deployment.** Fig. 10 summarises an end-to-end use of the pipeline with the costs measured on the ECG data. Training on about 6,822 30-s recordings took 48 s for the transform and 52 s for the eigendecomposition with two CPU threads, with a peak resident memory of 5.1 GB. The deployed model is the kernel parameters plus one weight matrix and a scaling vector, 253 kB in total, and classifying one recording, transform included, took a median of 10 ms on one CPU thread. Re-training after a batch of expert corrections needs neither a new transform nor a new decomposition, because neither depends on the labels; only the label-dependent steps are repeated.

![**Fig. 10** End-to-end use of the pipeline: training with a ranked list of suspicious labels for expert review, and on-line inference, with times, model size and memory measured on the ECG data](/home/user/rp_c_14/figures/fig_deployment.png){width=6.3in}

## 6.10 Sensitivity

Sensitivity was examined on a fixed random subset of 30 datasets outside the development set (seeds 0–2; full table in the supplement). Mean accuracy (%) on clean labels / at 40% symmetric / at 40% asymmetric noise is, for the reference setting, LOO-EM: 85.0 / 67.0 / 64.1; BIC penalty ×0.5: 84.8 / 66.9 / 65.2; BIC penalty ×2: 85.0 / 66.9 / 62.7; BIC penalty ×4: 85.1 / 66.7 / 62.7; prior strength 1: 84.9 / 66.6 / 64.1; prior strength 50: 84.7 / 66.6 / 62.9; 5 EM iterations: 85.0 / 66.9 / 64.2; 50 EM iterations: 85.0 / 67.1 / 64.1; wide-grid ridge: 85.2 / 65.9 / 62.2. The BIC penalty acts as described in Sect. 6.4: a smaller penalty buys accuracy under asymmetric noise with accuracy on clean labels. A strong prior on the transition matrix (50 pseudo-counts per class) pulls the full model towards the symmetric one and lowers accuracy under asymmetric noise; the number of EM iterations is immaterial within the range tried. Varying the number of kernels at 40% symmetric noise gives 1000 kernels: ridge 64.7, LOO-EM 65.5; 2500 kernels: ridge 65.5, LOO-EM 66.2; 5000 kernels: ridge 65.9, LOO-EM 66.9; 10,000 kernels: ridge 65.9, LOO-EM 67.0; the ordering of the methods does not depend on it.

## 6.11 Error analysis

Table 14 lists the datasets with the largest losses and gains of LOO-EM relative to wide-grid ridge, and Supplementary Fig. S4 plots the gain against training-set size. On clean labels the largest losses occur on Beef, PigCVP, EthanolLevel, ScreenType and Worms (losses of 3–7 points): problems on which the clean classifier is itself far from perfect (wide-grid ridge reaches 53–93% on them) or that have very few series per class (as few as 2), so that systematic confusions between similar classes look like label noise. LOO-EM changes the accuracy of 44% of datasets on clean labels at all. At 40% asymmetric noise the largest gains (15–19 points on TwoPatterns, LargeKitchenAppliances, SmoothSubspace, UWaveGestureLibraryAll and OSULeaf) occur on problems with 3–8 classes and at least 150 training series, where the classes are separable enough for the transition matrix to be identified; the largest losses (1–3 points) occur on 4 two-class problems and 1 with fewer than 60 training series. The per-dataset gain at 40% asymmetric noise correlates weakly with the training-set size (Spearman 0.20) and with clean accuracy (−0.15).

| Noise    | Group          | Dataset                     | n    | Classes   | Ridge (wide)   | LOO-EM (BIC)   | Diff.   |
|:---------|:---------------|:----------------------------|:-----|:----------|:---------------|:---------------|:--------|
| sym 0%   | largest losses | Beef                        | 30   | 5         | 77.3           | 70.7           | -6.7    |
| sym 0%   | largest losses | PigCVP                      | 104  | 52        | 92.9           | 88.9           | -3.9    |
| sym 0%   | largest losses | EthanolLevel                | 504  | 4         | 64.6           | 61.0           | -3.6    |
| sym 0%   | largest losses | ScreenType                  | 375  | 3         | 53.4           | 50.6           | -2.9    |
| sym 0%   | largest losses | Worms                       | 181  | 5         | 76.9           | 74.0           | -2.9    |
| sym 0%   | largest wins   | Lightning2                  | 60   | 2         | 71.8           | 74.1           | +2.3    |
| sym 0%   | largest wins   | Herring                     | 64   | 2         | 59.1           | 60.6           | +1.6    |
| sym 0%   | largest wins   | LargeKitchenAppliances      | 375  | 3         | 89.8           | 90.4           | +0.6    |
| sym 0%   | largest wins   | MiddlePhalanxOutlineCorrect | 600  | 2         | 84.1           | 84.5           | +0.4    |
| sym 0%   | largest wins   | Rock                        | 20   | 4         | 81.6           | 82.0           | +0.4    |
| asym 40% | largest losses | FaceFour                    | 24   | 4         | 53.6           | 50.5           | -3.2    |
| asym 40% | largest losses | FreezerSmallTrain           | 28   | 2         | 67.3           | 64.5           | -2.8    |
| asym 40% | largest losses | Wine                        | 57   | 2         | 53.0           | 50.7           | -2.2    |
| asym 40% | largest losses | MiddlePhalanxOutlineCorrect | 600  | 2         | 63.1           | 61.3           | -1.8    |
| asym 40% | largest losses | SemgHandGenderCh2           | 300  | 2         | 59.0           | 57.9           | -1.1    |
| asym 40% | largest wins   | TwoPatterns                 | 1000 | 4         | 72.5           | 91.9           | +19.4   |
| asym 40% | largest wins   | LargeKitchenAppliances      | 375  | 3         | 66.9           | 85.3           | +18.3   |
| asym 40% | largest wins   | SmoothSubspace              | 150  | 3         | 66.8           | 82.8           | +16.0   |
| asym 40% | largest wins   | UWaveGestureLibraryAll      | 896  | 8         | 79.5           | 95.4           | +16.0   |
| asym 40% | largest wins   | OSULeaf                     | 200  | 6         | 69.3           | 84.8           | +15.5   |

: **Table 14** Datasets with the largest losses and gains of LOO-EM (BIC) relative to wide-grid ridge (accuracy %, mean of five resamples)



# 7 Discussion

**What the real data changed.** On synthetic noise the ordering of remedies was clear: a wider penalty grid helped, LOO-EM added a further gain under class-conditional noise, cleaning cost accuracy at low noise, and backward correction was the worst method. On real annotation errors in 8,528 ECG recordings that ordering did not survive. The noise itself mattered little, no automatic relabelling or removal improved agreement with the expert labels and most reduced it, LOO-EM most clearly among the model-based methods, and the only correction that helped was the one that had performed worst on synthetic noise. Two features of the real setting explain most of this. First, the documented errors are few (4.8%) and concentrated in one direction, unreadable recordings labelled with a rhythm, whereas the classifier's own error rate is about 27%; when model error dominates label error, any procedure that treats disagreement between model and label as evidence against the label will mostly "correct" correct labels. Second, the classifier's errors are systematic, between two rhythm classes, and a full transition matrix is flexible enough to absorb a systematic confusion; the identifiability problem of Sect. 4.4 is therefore not a corner case but the regime that real data put us in. Synthetic benchmarks, including ours, inject noise into problems that the classifier solves well and draw the flips independently of the classifier's own confusions, which is the regime most favourable to correction. This agrees with the observation of Wei et al. (2024) that noisy-label methods for medical data are evaluated mostly with synthetic noise, and it argues for treating real-noise validation as a requirement rather than an addition.

**Practical recommendations.** (1) *Regularise, and select the penalty by the metric that matters.* Leave-one-out selection is free in closed form for any criterion. Squared error, the default in ROCKET-family implementations, favours majority classes; balanced error protected minority classes under symmetric noise on the UCR data (Sect. 6.7). Whether a wider grid helps depends on that criterion; on the ECG data, with the squared-error criterion, it did not. (2) *Do not relabel or remove training series automatically when the estimated noise is low or the classifier is weak.* On real data none of them gained accuracy and most lost it significantly; on synthetic data they paid off only at 30–40% noise. (3) *Use the suspicion ranking for review, not for automatic correction.* All scores, including the free closed-form LOO margin, concentrated real errors at the top of the list by a factor of about 1.7 over random; this makes expert review of the top of the list more productive than random re-annotation, but not reliable enough to act on unsupervised. (4) *Consider LOO-EM when the classifier is accurate and class-conditional noise is plausible and substantial*, as in the UCR experiments with 20–40% asymmetric noise and on ElectricDevices in the scaling experiment; report the model chosen by BIC and the number of labels it changes, and treat a change rate far above any plausible noise rate as a sign that the EM is explaining model error. (5) *Backward loss correction with arg-max anchors* is a cheap option when the noise is expected to be directed towards a known class; it was the only correction that helped on real data, but it was unstable under symmetric synthetic noise.

**Why regularisation goes so far, and where it stops.** Proposition 1 implies that under symmetric noise the ridge classifier needs no correction in expectation, only more averaging, and Corollary 1 shows that target smoothing, a popular remedy, is a no-op for this classifier. The interpolating regime created by ten thousand features and a few hundred series is where a narrow penalty grid prevents that averaging. The limit is that the averaging is chosen by a criterion; when the criterion and the evaluation metric disagree, as squared error and minority-class recall do, extra regularisation can hurt (Sects. 6.1, 6.6, 6.7).

**Relation to prior work.** The latent-label EM is classical (Dawid and Skene 1979; Bootkrajang and Kabán 2012); what is specific here is that the predictive distribution is the closed-form LOO one, which removes self-confirmation in the $p\gg n$ regime (the in-sample ablation is inert) and makes each iteration a matrix product, and that the noise model is selected rather than assumed. Prevalidated ridge regression (Dempster et al. 2024b) uses the same LOO identity for calibration on clean labels. Of the time-series-specific noisy-label methods we ran only CTW (Ma et al. 2023), on its authors' benchmark datasets, and we make no claim relative to the others. The penalty criterion is free, LOO-EM costs a small multiple of the eigendecomposition and less than 5-fold confident learning, and the deployed model is the same linear classifier in every variant, so whether LOO-EM is worth running is decided by the data rather than by its cost.

# 8 Limitations

*One real-noise dataset.* The real-world evaluation rests on one ECG dataset, whose revised labels are themselves imperfect (low inter-expert agreement on contentious recordings) and whose training recordings were mostly not re-read individually, so undetected errors remain in the reference. We looked for further public time-series data with paired original and verified labels; the multi-scorer sleep datasets we identified were no longer downloadable. Industrial condition-monitoring data with technician labels and maintenance-verified ground truth, a natural next test, were not available to us. *Synthetic noise.* All UCR, UEA and scaling results use injected noise, which on the evidence of Sect. 6.6 is more favourable to label correction than real noise. *One archive.* The UCR and UEA archives contain many small, pre-segmented problems, some with train–test duplicates and with series cut from the same recordings; absolute accuracies are archive-specific. *Design history.* The proposed variant was designed after an earlier one failed; the confirmation set guards against the resulting optimism, but it contains only 60 datasets, and several individual comparisons on it are not significant. The penalty-criterion analysis is exploratory. *Baselines.* Only one time-series-specific deep method (CTW, re-implemented, on seven datasets) and simple deep baselines on six small datasets were run, on CPU and with published rather than tuned hyper-parameters; MultiRocket features were planned but not run. *Weak base classifier on the ECG data.* MiniRocket with ridge regression is far from the best published ECG rhythm classifiers; with a stronger classifier the balance between model error and label error, and hence the value of correction, could differ. *Scale.* The decomposition limits the dual route to roughly 10^4^ training series on a workstation; the primal route scales linearly in $n$ for a fixed feature dimension. *Class-conditional model.* LOO-EM assumes noise that depends only on the class; the instance-dependent extension we tested was rejected. *Identifiability.* Without outside knowledge, class-conditional noise cannot be told from systematic model error, and no choice of penalty in the BIC removes that; Sect. 6.6 shows the consequence on real data.

# 9 Conclusion

For ridge classifiers on random convolutional kernel features, the first response to suspected label noise should be regularisation chosen with the evaluation metric in mind, not automatic relabelling. Symmetric noise leaves the expected decision function order-preserving and inflates its variance, so regularisation addresses it directly, and label smoothing provably does nothing. On 108 UCR datasets with synthetic noise a wider penalty grid recovered up to 4.2 accuracy points, and LOO-EM, an inexpensive expectation–maximisation procedure on closed-form leave-one-out predictions, added a further gain under heavy class-conditional noise. On real annotation errors in 8,528 ECG recordings, however, training on the noisy labels cost almost nothing, no automatic relabelling or removal method improved agreement with the expert-revised labels and most, LOO-EM included, reduced it, and label-error rankings were useful only as a guide for review. The result that transfers is the practical one in the title, and the result that does not is the size of the benefit of label correction measured on synthetic noise. Future noisy-label methods for time series should be validated on data with documented real errors, and the leave-one-out machinery offers a cheap way to rank labels for the expert review that such data require.

# Declarations

**Funding** [To be completed by the authors.]

**Competing interests** [To be completed by the authors; e.g. "The authors have no competing interests to declare that are relevant to the content of this article."]

**Ethics approval and consent** Not applicable. The study uses only public benchmark datasets and involves no human participants or animals.

**Data availability** The UCR and UEA datasets are publicly available from the time series classification archive (https://www.timeseriesclassification.com). The ECG recordings and all four versions of their labels are publicly available from PhysioNet (https://physionet.org/content/challenge-2017/1.0.0/) under the Open Data Commons Attribution License v1.0. All data are downloaded automatically by the accompanying code; no new data were collected.

**Code availability** The complete source code, configuration, raw per-run results, and the scripts that generate every table and figure in this article are available at [repository URL to be inserted by the authors].

**Author contributions** [To be completed by the authors.]

**Use of AI tools** [To be completed by the authors in line with the publisher's policy on the disclosure of AI assistance in the preparation of the code, experiments and manuscript.]

# References

Angluin D, Laird P (1988) Learning from noisy examples. Mach Learn 2:343–370. https://doi.org/10.1023/A:1022873112823

Atkinson G, Metsis V (2020) Identifying label noise in time-series datasets. In: Adjunct proceedings of the 2020 ACM international joint conference on pervasive and ubiquitous computing (UbiComp-ISWC '20). ACM, pp 238–243. https://doi.org/10.1145/3410530.3414366

Bagnall A, Lines J, Bostrom A, Large J, Keogh E (2017) The great time series classification bake off: a review and experimental evaluation of recent algorithmic advances. Data Min Knowl Discov 31:606–660. https://doi.org/10.1007/s10618-016-0483-9

Bartlett PL, Long PM, Lugosi G, Tsigler A (2020) Benign overfitting in linear regression. Proc Natl Acad Sci USA 117:30063–30070. https://doi.org/10.1073/pnas.1907378117

Bootkrajang J, Kabán A (2012) Label-noise robust logistic regression and its applications. In: Machine learning and knowledge discovery in databases (ECML PKDD 2012). Lecture notes in computer science, vol 7523. Springer, pp 143–158. https://doi.org/10.1007/978-3-642-33460-3_15

Brier GW (1950) Verification of forecasts expressed in terms of probability. Mon Weather Rev 78:1–3. https://doi.org/10.1175/1520-0493(1950)078<0001:VOFEIT>2.0.CO;2

Brodley CE, Friedl MA (1999) Identifying mislabeled training data. J Artif Intell Res 11:131–167. https://doi.org/10.1613/jair.606

Castellani A, Schmitt S, Hammer B (2021) Estimating the electrical power output of industrial devices with end-to-end time-series classification in the presence of label noise. In: Machine learning and knowledge discovery in databases. Research track (ECML PKDD 2021). Lecture notes in computer science, vol 12975. Springer, pp 469–484. https://doi.org/10.1007/978-3-030-86486-6_29

Chen J, Cao T, Xu J, Li J, Chen Z, Xiao T, Yang Y (2024) Con4m: context-aware consistency learning framework for segmented time series classification. arXiv preprint arXiv:2408.00041

Chen P, Ye J, Chen G, Zhao J, Heng PA (2021) Robustness of accuracy metric and its inspirations in learning with noisy labels. Proc AAAI Conf Artif Intell 35(13):11451–11461. https://doi.org/10.1609/aaai.v35i13.17364

Clifford GD, Liu C, Moody B, Lehman LH, Silva I, Li Q, Johnson AEW, Mark RG (2017) AF classification from a short single lead ECG recording: the PhysioNet/Computing in Cardiology Challenge 2017. In: Computing in Cardiology, vol 44. https://doi.org/10.22489/CinC.2017.065-469

Dau HA, Bagnall A, Kamgar K, Yeh CCM, Zhu Y, Gharghabi S, Ratanamahatana CA, Keogh E (2019) The UCR time series archive. IEEE/CAA J Autom Sin 6:1293–1305. https://doi.org/10.1109/JAS.2019.1911747

Dawid AP, Skene AM (1979) Maximum likelihood estimation of observer error-rates using the EM algorithm. J R Stat Soc Ser C Appl Stat 28:20–28. https://doi.org/10.2307/2346806

Dempster A, Petitjean F, Webb GI (2020) ROCKET: exceptionally fast and accurate time series classification using random convolutional kernels. Data Min Knowl Discov 34:1454–1495. https://doi.org/10.1007/s10618-020-00701-z

Dempster A, Schmidt DF, Webb GI (2021) MiniRocket: a very fast (almost) deterministic transform for time series classification. In: Proceedings of the 27th ACM SIGKDD conference on knowledge discovery and data mining. ACM, pp 248–257. https://doi.org/10.1145/3447548.3467231

Dempster A, Schmidt DF, Webb GI (2023) Hydra: competing convolutional kernels for fast and accurate time series classification. Data Min Knowl Discov 37:1779–1805. https://doi.org/10.1007/s10618-023-00939-3

Dempster A, Schmidt DF, Webb GI (2024a) quant: a minimalist interval method for time series classification. Data Min Knowl Discov 38:2377–2402. https://doi.org/10.1007/s10618-024-01036-9

Dempster A, Webb GI, Schmidt DF (2024b) Prevalidated ridge regression is a highly-efficient drop-in replacement for logistic regression for high-dimensional data. arXiv preprint arXiv:2401.15610

Demšar J (2006) Statistical comparisons of classifiers over multiple data sets. J Mach Learn Res 7:1–30

Efron B (1979) Bootstrap methods: another look at the jackknife. Ann Stat 7:1–26. https://doi.org/10.1214/aos/1176344552

Frénay B, Verleysen M (2014) Classification in the presence of label noise: a survey. IEEE Trans Neural Netw Learn Syst 25:845–869. https://doi.org/10.1109/TNNLS.2013.2292894

Ghosh A, Kumar H, Sastry PS (2017) Robust loss functions under label noise for deep neural networks. Proc AAAI Conf Artif Intell 31(1). https://doi.org/10.1609/aaai.v31i1.10894

Goldberger AL, Amaral LAN, Glass L, Hausdorff JM, Ivanov PC, Mark RG, Mietus JE, Moody GB, Peng CK, Stanley HE (2000) PhysioBank, PhysioToolkit, and PhysioNet: components of a new research resource for complex physiologic signals. Circulation 101:e215–e220. https://doi.org/10.1161/01.CIR.101.23.e215

Golub GH, Heath M, Wahba G (1979) Generalized cross-validation as a method for choosing a good ridge parameter. Technometrics 21:215–223. https://doi.org/10.1080/00401706.1979.10489751

Guo C, Pleiss G, Sun Y, Weinberger KQ (2017) On calibration of modern neural networks. In: Proceedings of the 34th International Conference on Machine Learning (ICML). arXiv:1706.04599

Han B, Yao Q, Yu X, Niu G, Xu M, Hu W, Tsang I, Sugiyama M (2018) Co-teaching: robust training of deep neural networks with extremely noisy labels. In: Advances in neural information processing systems 31 (NeurIPS 2018). arXiv:1804.06872

Hastie T, Tibshirani R, Friedman J (2009) The elements of statistical learning: data mining, inference, and prediction, 2nd edn. Springer, New York. https://doi.org/10.1007/978-0-387-84858-7

Hoerl AE, Kennard RW (1970) Ridge regression: biased estimation for nonorthogonal problems. Technometrics 12:55–67. https://doi.org/10.1080/00401706.1970.10488634

Holm S (1979) A simple sequentially rejective multiple test procedure. Scand J Stat 6:65–70

Ismail Fawaz H, Forestier G, Weber J, Idoumghar L, Muller PA (2019) Deep learning for time series classification: a review. Data Min Knowl Discov 33:917–963. https://doi.org/10.1007/s10618-019-00619-1

Ismail Fawaz H, Lucas B, Forestier G, Pelletier C, Schmidt DF, Weber J, Webb GI, Idoumghar L, Muller PA, Petitjean F (2020) InceptionTime: finding AlexNet for time series classification. Data Min Knowl Discov 34:1936–1962. https://doi.org/10.1007/s10618-020-00710-y

Li Y, Luo L, Aickelin U (2025) Dynamical label augmentation and calibration for noisy electronic health records. arXiv preprint arXiv:2505.07320

Li Z, Liu Y, Tanaka G (2026) Robust and efficient noisy-label time-series classification via dynamic time warping based granular ball computing. arXiv preprint arXiv:2608.11704

Liu J, Yang K, He J, Yang C, Yang S, Zhou Y (2026) Robust learning with time series noisy labels via self-supervised learning and soft labels refurbishment. Neurocomputing 676:133059. https://doi.org/10.1016/j.neucom.2026.133059

Liu T, Tao D (2016) Classification with noisy labels by importance reweighting. IEEE Trans Pattern Anal Mach Intell 38:447–461. https://doi.org/10.1109/TPAMI.2015.2456899

Liu Z, Ma P, Chen D, Pei W, Ma Q (2023) Scale-teaching: robust multi-scale training for time series classification with noisy labels. In: Advances in neural information processing systems 36 (NeurIPS 2023)

Lukasik M, Bhojanapalli S, Menon AK, Kumar S (2020) Does label smoothing mitigate label noise? In: Proceedings of the 37th International Conference on Machine Learning (ICML). arXiv:2003.02819

Ma P, Liu Z, Zheng J, Wang L, Ma Q (2023) CTW: confident time-warping for time-series label-noise learning. In: Proceedings of the thirty-second international joint conference on artificial intelligence (IJCAI-23), pp 4046–4054. https://doi.org/10.24963/ijcai.2023/450

McNemar Q (1947) Note on the sampling error of the difference between correlated proportions or percentages. Psychometrika 12:153–157. https://doi.org/10.1007/BF02295996

Middlehurst M, Ismail-Fawaz A, Guillaume A, Holder C, Guijo-Rubio D, Bulatova G, Tsaprounis L, Mentel L, Walter M, Schäfer P, Bagnall A (2024b) aeon: a Python toolkit for learning from time series. arXiv preprint arXiv:2406.14231

Middlehurst M, Schäfer P, Bagnall A (2024a) Bake off redux: a review and experimental evaluation of recent time series classification algorithms. Data Min Knowl Discov 38:1958–2031. https://doi.org/10.1007/s10618-024-01022-1

Nagaraj S, Gerych W, Tonekaboni S, Goldenberg A, Ustun B, Hartvigsen T (2025) Learning under temporal label noise. In: International Conference on Learning Representations (ICLR). arXiv:2402.04398

Natarajan N, Dhillon IS, Ravikumar P, Tewari A (2013) Learning with noisy labels. In: Advances in neural information processing systems 26 (NIPS 2013)

Northcutt C, Jiang L, Chuang I (2021) Confident learning: estimating uncertainty in dataset labels. J Artif Intell Res 70:1373–1411. https://doi.org/10.1613/jair.1.12125

Patrini G, Rozza A, Menon AK, Nock R, Qu L (2017) Making deep neural networks robust to label noise: a loss correction approach. In: 2017 IEEE conference on computer vision and pattern recognition (CVPR), pp 2233–2241. https://doi.org/10.1109/CVPR.2017.240

Pollard T, Moody BE, Lehman LH, Gow BJ, Fernandes C, Xie C, Johnson A, Mark RG, et al. (2026) PhysioNet as a global platform for biomedical research. Nat Health 1:792–795. https://doi.org/10.1038/s44360-026-00096-z

Schwarz G (1978) Estimating the dimension of a model. Ann Stat 6:461–464. https://doi.org/10.1214/aos/1176344136

Song H, Kim M, Park D, Shin Y, Lee JG (2023) Learning from noisy labels with deep neural networks: a survey. IEEE Trans Neural Netw Learn Syst 34:8135–8153. https://doi.org/10.1109/TNNLS.2022.3152527

Tan CW, Dempster A, Bergmeir C, Webb GI (2022) MultiRocket: multiple pooling operators and transformations for fast and effective time series classification. Data Min Knowl Discov 36:1623–1646. https://doi.org/10.1007/s10618-022-00844-1

Teijeiro T, García CA, Castro D, Félix P (2017) Arrhythmia classification from the abductive interpretation of short single-lead ECG records. In: Computing in Cardiology, vol 44. https://doi.org/10.22489/CinC.2017.166-054

Tibshirani RJ, Efron B (2002) Pre-validation and inference in microarrays. Stat Appl Genet Mol Biol 1(1):Article 1. https://doi.org/10.2202/1544-6115.1000

Wang Z, Yan W, Oates T (2017) Time series classification from scratch with deep neural networks: a strong baseline. In: 2017 international joint conference on neural networks (IJCNN), pp 1578–1585. https://doi.org/10.1109/IJCNN.2017.7966039

Wei Y, Deng Y, Sun C, Lin M, Jiang H, Peng Y (2024) Deep learning with noisy labels in medical prediction problems: a scoping review. arXiv preprint arXiv:2403.13111

Wilcoxon F (1945) Individual comparisons by ranking methods. Biometrics Bull 1:80–83. https://doi.org/10.2307/3001968

Wilson DL (1972) Asymptotic properties of nearest neighbor rules using edited data. IEEE Trans Syst Man Cybern SMC-2:408–421. https://doi.org/10.1109/TSMC.1972.4309137

Xia X, Liu T, Han B, Wang N, Gong M, Liu H, Niu G, Tao D, Sugiyama M (2020) Part-dependent label noise: towards instance-dependent label noise. In: Advances in Neural Information Processing Systems 33 (NeurIPS 2020). arXiv:2006.07836

Zhang Z, Sabuncu MR (2018) Generalized cross entropy loss for training deep neural networks with noisy labels. In: Advances in neural information processing systems 31 (NeurIPS 2018). arXiv:1805.07836
