---
title: "Regularise before you relabel: label noise in random convolutional kernel time series classification"
author: "[Author One]^1\\*^, [Author Two]^2^"
---

^1^ [Department, Institution, City, Country] · ^2^ [Department, Institution, City, Country]

\* Corresponding author: [name], [e-mail address], ORCID [0000-0000-0000-0000]

## Abstract

Random convolutional kernel transforms followed by a ridge classifier (ROCKET, MiniRocket and successors) are fast and accurate time series classifiers, but their behaviour under wrong training labels has not been studied systematically. We evaluate MiniRocket with ridge regression on 108 UCR datasets, five resamples each, under symmetric, asymmetric and instance-dependent label noise of up to 40%. First, part of the fragility is an artefact of the customary penalty grid: at 30% symmetric noise the selected penalty sits on the grid's upper end in 98% of runs, and widening the grid raises mean accuracy by up to 4.2 points with no loss on clean labels. We prove that symmetric noise leaves the expected ridge decision function order-preserving and only inflates its variance. The gain is metric-dependent: under symmetric noise it does not extend to balanced accuracy or macro-F1. Second, cleaning pipelines on the same features (confident learning, edited nearest neighbours, hard relabelling) detect mislabelled series well yet lower clean-data accuracy by 2.1–6.7 points; calibrated confident learning pays off only at 30–40% symmetric noise. Third, we propose LOO-EM, an expectation–maximisation procedure for class-conditional noise that always predicts with closed-form leave-one-out scores, reuses a single eigendecomposition and selects the noise model by BIC. On 60 confirmation datasets it improves on wide-grid ridge by 1.3 points at 30% symmetric and 2.2 points at 40% asymmetric noise, costs 0.3 points on clean labels, and gains nothing under instance-dependent noise. An earlier variant that failed on many-class data is reported.

**Keywords** Time series classification · Label noise · Random convolutional kernels · Ridge regression · Leave-one-out cross-validation · Expectation–maximisation

# 1 Introduction

Time series classification (TSC) underlies activity recognition, condition monitoring and physiological signal analysis. Among the many algorithm families that have been benchmarked on the UCR archive (Dau et al. 2019; Bagnall et al. 2017; Middlehurst et al. 2024a), methods that transform each series with a large number of random convolutional kernels and train a linear ridge classifier on the pooled outputs occupy a distinctive position: ROCKET (Dempster et al. 2020), MiniRocket (Dempster et al. 2021), MultiRocket (Tan et al. 2022) and Hydra (Dempster et al. 2023) approach the accuracy of much more expensive ensembles and deep networks (Ismail Fawaz et al. 2019; Ismail Fawaz et al. 2020) while training in seconds on a CPU. This makes them a natural default in applied work.

Applied labels are rarely perfect. Sensor recordings are annotated by fatigued operators, by weak heuristics, or by propagating a label over a whole recording session, and the resulting errors can be random or systematic (Frénay and Verleysen 2014; Atkinson and Metsis 2020). Research on label noise in TSC has concentrated on deep networks: self-supervised and ensembling schemes (Castellani et al. 2021), confident-sample augmentation (Ma et al. 2023) and multi-scale co-teaching (Liu et al. 2023), with distance-based alternatives appearing only recently (Li et al. 2026). For the random-kernel family we found no systematic study, although its training regime is unusual: roughly ten thousand features are fitted to training sets that often contain fewer than a few hundred series, so the ridge classifier operates close to interpolation, the regime in which fitting noise can be either benign or harmful depending on the regularisation (Bartlett et al. 2020).

This paper asks four questions, each tied to an experiment.

**RQ1.** How much accuracy do random-kernel ridge classifiers lose under symmetric, asymmetric and instance-dependent label noise, and how much of the loss is caused by the regularisation protocol rather than by the noise itself?

**RQ2.** Do generic label-cleaning strategies (confident learning, nearest-neighbour editing, relabelling from out-of-sample predictions) help when applied to these features?

**RQ3.** Can label correction be made essentially free by exploiting the closed-form leave-one-out (LOO) predictions of ridge regression, and does it improve on a properly regularised ridge classifier?

**RQ4.** Do the answers hold for other transforms, for multivariate data, for larger training sets and relative to other classifier families?

Our contributions are the following.

1. *A large-scale characterisation.* MiniRocket with ridge regression is evaluated on 108 UCR datasets with five resamples and twelve noise conditions (144,785 recorded runs for 23 method variants), with secondary experiments on Hydra features, 18 multivariate UEA datasets, three of the four largest UCR datasets, and non-kernel and deep baselines.
2. *An explanation and a one-line fix.* We prove that for ridge classification symmetric label noise scales the expected decision function without changing its arg-max and adds variance that only the penalty can control, whereas asymmetric noise shifts the arg-max itself. Empirically the default penalty grid of ROCKET pipelines is too narrow under noise; widening it is the largest single improvement in accuracy that we observed, with the qualification that under symmetric noise the improvement is confined to accuracy and is absent or negative for class-balanced metrics.
3. *A mostly negative result on cleaning.* Detection of mislabelled series from closed-form LOO scores is as accurate as from explicit 5-fold cross-validation, but acting on the detections by removal or hard relabelling costs accuracy on clean and mildly noisy data; confident learning helps only at 30–40% symmetric noise, and only when its probabilities are calibrated.
4. *LOO-EM.* An EM algorithm for class-conditional noise in which the E-step uses closed-form LOO predictive probabilities and the M-step is a ridge refit on soft targets. Because the hat matrix does not depend on the labels, the whole procedure reuses one eigendecomposition. The noise model (none, symmetric or full) is selected by BIC.
5. *A documented failure and a trade-off.* The first frozen version of LOO-EM, which always used a full transition matrix, lost accuracy on held-out many-class data. We report it, explain it, and show that its remaining advantage under heavy asymmetric noise cannot be had without the corresponding risk on clean data.

# 2 Related work

**Random convolutional kernel classifiers.** ROCKET convolves each series with 10,000 random kernels and pools each output by its maximum and its proportion of positive values (Dempster et al. 2020). MiniRocket fixes most of the randomness and keeps only the proportion-of-positive-values pooling (Dempster et al. 2021); MultiRocket adds first differences and three further pooling operators (Tan et al. 2022); Hydra counts which kernel in each group responds most strongly (Dempster et al. 2023). All of them train a ridge classifier whose penalty is chosen by leave-one-out cross-validation over ten values between 10^−3^ and 10^3^, and this grid is the default in widely used implementations (Middlehurst et al. 2024b). Recent benchmark studies place these methods among the most accurate fast classifiers (Middlehurst et al. 2024a); QUANT obtains comparable accuracy from interval quantiles and extremely randomised trees (Dempster et al. 2024a). The closest methodological relative of our work is prevalidated ridge regression (Dempster et al. 2024b), which uses closed-form LOO predictions, in the spirit of pre-validation (Tibshirani and Efron 2002), to scale ridge outputs into calibrated probabilities. It targets probability calibration on clean labels; it does not address label noise.

**Learning with label noise.** Learning from randomly mislabelled examples has been studied since Angluin and Laird (1988). Surveys distinguish noise that is independent of the features given the class (symmetric or asymmetric class-conditional noise) from instance-dependent noise (Frénay and Verleysen 2014; Song et al. 2023). Three families of remedies are relevant here. *Noise-robust losses and loss correction* modify the objective, for example through the generalised cross-entropy (Zhang and Sabuncu 2018), loss conditions for robustness (Ghosh et al. 2017), importance reweighting (Liu and Tao 2016) or forward and backward correction with an estimated transition matrix (Natarajan et al. 2013; Patrini et al. 2017). *Latent-label models* treat the true label as a hidden variable and estimate a transition matrix by EM, from the annotator model of Dawid and Skene (1979) to noise-robust logistic regression (Bootkrajang and Kabán 2012). *Data cleaning* identifies and removes or relabels suspicious samples, from edited nearest neighbours (Wilson 1972) and classification filters (Brodley and Friedl 1999) to confident learning (Northcutt et al. 2021), and sample-selection schemes such as Co-teaching (Han et al. 2018). Chen et al. (2021) showed that under symmetric noise accuracy measured on noisy validation labels is an affine function of clean accuracy, which justifies model selection on noisy labels; our Proposition 1 is the analogous statement for the ridge decision function itself.

**Label noise in time series.** Atkinson and Metsis (2020) studied the identification of mislabelled sensor segments. Castellani et al. (2021), Ma et al. (2023) and Liu et al. (2023) proposed deep networks for noisy-label TSC evaluated on subsets of the UCR and UEA archives; Li et al. (2026) proposed a granular-ball method built on dynamic time warping. None of these works evaluates random-kernel classifiers, examines the role of the ridge penalty, or uses the closed-form LOO structure.

**Research gap.** What exists is (i) fast kernel-transform classifiers evaluated on clean benchmarks, (ii) generic and deep-learning-specific noise remedies, and (iii) a closed-form LOO calibration method for clean data. What is missing is knowledge of how the kernel-transform family behaves under label noise, whether generic cleaning helps it, and whether its algebraic structure can be exploited. This matters because these classifiers are used precisely where compute is scarce and labels are cheap, and therefore noisy.

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

![**Fig. 1** Overview of LOO-EM. The eigendecomposition of the centred Gram matrix is computed once; every EM iteration and the final fit reuse it](/home/user/rp_c_14/figures/fig_method_overview.png){width=6.3in}

## 4.3 Cost

The eigendecomposition costs $O(n^3)$ (or $O(np^2)$ through the primal when $n>p$) and is needed by any LOO-tuned ridge classifier. Each EM iteration costs $O(n^2C)$ per candidate penalty; the hat diagonal for each penalty is label-independent and cached, and after the first iteration only penalties adjacent to the current one are examined, extending the window while the optimum lies on its edge. No model is ever refitted from scratch. In contrast, k-fold cleaning needs $k$ further eigendecompositions.

## 4.4 What the likelihood cannot decide

Equation (5) cannot by itself distinguish heavy class-conditional noise from genuinely overlapping classes: a clean-posterior model with large off-diagonal $T$ and a flat predictor with $T=I$ can reach the same likelihood. Any procedure that corrects aggressively when predictions and labels disagree will therefore also "correct" hard but correctly labelled data. BIC resolves the ambiguity conservatively. Section 6.4 quantifies the price of that choice.

# 5 Experimental setup

**Data.** The 112 equal-length univariate problems of the UCR archive (Dau et al. 2019) were obtained through the aeon toolkit (Middlehurst et al. 2024b). The 108 problems with fewer than 3,000 training series form the main benchmark (training sizes 16–1,800, median 180; 2–60 classes; Fig. 2); of the four larger ones, FordA, FordB and Crop (3,601–7,200 training series) are used for a separate scalability experiment without the 5-fold baselines, which are too expensive there; ElectricDevices (8,926 series) could not be completed on the available machine and is not reported. Generalisation to multivariate data uses the equal-length UEA problems that fit the available machine (18 datasets).

![**Fig. 2** Training-set size, number of classes and series length of the UCR datasets used](/home/user/rp_c_14/figures/fig_dataset_distribution.png){width=6.3in}

**Resampling and seeds.** Five seeds (0–4) were fixed in advance. Seed 0 is the archive split; seeds 1–4 are stratified resamples of the pooled data with the original training and test sizes. Secondary experiments use seeds 0–2. Per-dataset scores are means over seeds.

**Noise.** Noise is applied to training labels only; test labels are always clean. *Symmetric*: a fraction $\rho\in\{0,0.1,0.2,0.3,0.4\}$ of labels is replaced by a uniformly chosen different class. *Asymmetric*: a fraction $\rho\in\{0.1,\dots,0.4\}$ of each class $c$ is relabelled $c+1$ (mod $C$). *Boundary*: the fraction $\rho\in\{0.1,0.2,0.3\}$ of training series with the smallest clean LOO margin is relabelled to the strongest competing class; this instance-dependent model is adversarial to margin-based detectors by construction. Each noise realisation is a deterministic function of dataset, seed, noise model and rate, so all methods see identical labels. Conditions in which a class disappears from the noisy training set are skipped for that dataset and seed.

**Features and classifier.** MiniRocket with 10,000 kernels as implemented in aeon, fitted on training series only; columns standardised with training statistics; ridge classifier as in Sect. 3.1. Our implementation reproduces scikit-learn's `RidgeClassifierCV` (identical selected penalty and predictions in a unit test). The *default grid* is ten log-spaced penalties in [10^−3^, 10^3^]; the *wide grid* is 25 log-spaced penalties in [10^−3^, 10^9^].

**Methods compared on identical features and noise.** (i) Ridge with the default grid; (ii) ridge with the wide grid; (iii) LOO-EM with BIC selection (proposed); (iv) LOO-EM v1, the earlier frozen variant with a full transition matrix accepted through a one-parameter gate; (v) confident learning (Northcutt et al. 2021) using the `cleanlab` implementation with 5-fold out-of-sample ridge scores passed through a softmax whose temperature is fitted by maximum likelihood on those scores, flagged series removed (the uncalibrated variant with unit temperature, which we ran first, is kept in the tables because the difference is instructive); (vi) relabelling from the same 5-fold predictions; (vii) edited nearest neighbours with $k=3$ in the standardised feature space (Wilson 1972); (viii) one-round relabelling and (ix) removal of series whose LOO prediction disagrees with their label. Methods (iii)–(ix) use the wide grid. Other classifier families trained on the same noisy labels: QUANT (Dempster et al. 2024a) and 1-nearest-neighbour with Euclidean distance, and, on six small datasets, a fully convolutional network (Wang et al. 2017) trained with cross-entropy, with the generalised cross-entropy loss (Zhang and Sabuncu 2018) and with Co-teaching (Han et al. 2018) given the true noise rate. Deviations from original implementations: the deep baselines are trained for 100 epochs with Adam (learning rate 10^−3^, batch size up to 32) without early stopping, since no clean validation labels exist.

**Protocol and the two design stages.** Method design used 12 development datasets drawn at random. The resulting configuration (v1) was frozen and run on all datasets. An interim inspection of the first completed datasets showed that v1 lost accuracy on clean, many-class data (Sect. 6.4). We then designed the BIC variant on a *design set* of 39 datasets and froze it before any other result was inspected. The 60 datasets that belong to neither the development nor the design set form the *confirmation set* on which the primary comparison is made. An anchor-point variant (Patrini et al. 2017) tried at the same stage was worse than wide-grid ridge on the design set and was dropped. The full history is in the supplementary protocol.

**Metrics and statistics.** The primary metric is test accuracy. Balanced accuracy and macro-F1 are reported alongside it where they change a conclusion (Sect. 6.1) and in full in the supplement. For label-error detection we report the AUROC of each method's suspicion score and the fraction of training labels that are correct after cleaning. Methods are compared by two-sided Wilcoxon signed-rank tests (Wilcoxon 1945) over datasets (unit of analysis: the dataset, as recommended by Demšar 2006), with Holm correction over the twelve noise conditions (Holm 1979), the matched-pairs rank-biserial correlation as effect size, and percentile bootstrap 95% confidence intervals (10,000 resamples of datasets) for the mean difference. The test is chosen because per-dataset accuracy differences are paired, bounded and not normally distributed.

**Leakage audit.** Transforms, scaling, penalty selection and all EM quantities use training data only. A duplicate scan of the archive splits found exact train–test duplicates in 16 datasets; 8 datasets have more than 1% duplicated test series (Crop, ElectricDevices, InlineSkate, InsectEPGSmallTrain, PowerCons, ToeSegmentation1, Worms, WormsTwoClass). This affects all methods equally; the main comparison is repeated without these datasets in the supplement and is unchanged in direction.

**Implementation.** Python 3.13, aeon 1.6, scikit-learn 1.9, NumPy/SciPy, cleanlab 2.9, PyTorch 2.15 (CPU). All experiments ran on a four-core CPU machine with 15 GB of memory; no GPU was used.

# 6 Results

## 6.1 RQ1: fragility, and how much of it is the penalty grid

Table 1 and Fig. 3 give mean test accuracy over the 108 datasets. With the default grid, accuracy falls from 87.1% on clean labels to 79.1% at 20% and 64.1% at 40% symmetric noise, and to 59.6% at 40% asymmetric noise.

| Method                            | S 0   | S 10   | S 20   | S 30   | S 40   | A 10   | A 20   | A 30   | A 40   | B 10   | B 20   | B 30   |
|:----------------------------------|:------|:-------|:-------|:-------|:-------|:-------|:-------|:-------|:-------|:-------|:-------|:-------|
| Ridge (default grid)              | 87.1  | 83.6   | 79.1   | 72.2   | 64.1   | 84.1   | 79.6   | 70.4   | 59.6   | 83.0   | 79.2   | 74.7   |
| Ridge (wide grid)                 | 87.2  | 83.9   | 79.6   | 73.6   | 65.8   | 84.5   | 81.3   | 73.5   | 63.8   | 83.2   | 80.1   | 76.9   |
| LOO-EM (BIC, proposed)            | 86.8  | 84.1   | 80.2   | 74.8   | 66.8   | 84.7   | 81.9   | 74.1   | 65.8   | 83.1   | 80.0   | 76.9   |
| LOO-EM v1 (full T)                | 85.4  | 82.4   | 78.6   | 73.5   | 65.8   | 82.7   | 80.2   | 74.0   | 67.4   | 82.8   | 79.9   | 76.4   |
| Confident learning (calibrated)   | 85.0  | 83.0   | 79.9   | 75.1   | 67.1   | 83.4   | 81.1   | 74.0   | 64.5   | 82.9   | 80.7   | 77.7   |
| Confident learning (uncalibrated) | 83.7  | 82.0   | 78.0   | 72.6   | 64.8   | 82.0   | 79.4   | 72.8   | 63.7   | 82.5   | 80.5   | 77.4   |
| 5-fold relabel                    | 83.2  | 80.6   | 76.3   | 70.8   | 63.8   | 80.8   | 78.0   | 71.4   | 63.1   | 82.1   | 80.1   | 77.4   |
| ENN                               | 80.5  | 78.9   | 76.8   | 72.6   | 64.9   | 79.1   | 76.9   | 70.8   | 60.4   | 79.4   | 79.5   | 76.3   |
| LOO relabel (1 round)             | 82.6  | 80.4   | 77.0   | 71.3   | 64.2   | 80.7   | 78.4   | 71.2   | 62.3   | 82.1   | 80.7   | 78.3   |
| LOO remove                        | 84.1  | 82.2   | 79.1   | 73.5   | 66.2   | 82.5   | 80.7   | 73.9   | 65.5   | 82.8   | 80.7   | 78.1   |

: **Table 1** Mean test accuracy (%) over 108 UCR datasets (five resamples each) on MiniRocket features. For each column only datasets on which the condition is defined for all methods are averaged. S, A and B denote symmetric, asymmetric and boundary noise at the given rate (%)


![**Fig. 3** Mean test accuracy over datasets as a function of the noise rate for the three noise models](/home/user/rp_c_14/figures/fig_accuracy_vs_noise.png){width=6.3in}

The selected penalty tells the story predicted by Proposition 1 (Fig. 4). On clean labels the default grid's upper end is selected in 34% of runs; at 20% symmetric noise in 96% and at 30% in 98%. With the wide grid, 95% of runs at 30% noise select a penalty above 10^3^.

![**Fig. 4** Distribution of the LOO-selected penalty under symmetric noise for the default and the wide grid](/home/user/rp_c_14/figures/fig_selected_alpha.png){width=6.3in}

Widening the grid leaves clean accuracy unchanged (+0.1 points (95% CI −0.1 to +0.2; Holm-adjusted p = 0.648; 21 wins, 32 losses)) and helps under every noise model (Table 2): +1.4 points (95% CI +0.3 to +2.5; Holm-adjusted p = 0.011; 70 wins, 36 losses) at 30% symmetric, +4.2 points (95% CI +2.6 to +5.9; Holm-adjusted p < 0.001; 80 wins, 28 losses) at 40% asymmetric and +2.2 points (95% CI +1.5 to +3.0; Holm-adjusted p < 0.001; 71 wins, 17 losses) at 30% boundary noise. The gain under asymmetric noise is larger than under symmetric noise, although regularisation cannot remove the bias of Proposition 2: it removes the variance component, which is present under every noise model. The benefit is not uniform. For the 36 datasets with fewer than 100 training series the wide grid is worse than the default at 30% symmetric noise (68.5% versus 70.5%), where the LOO criterion itself is computed from a few dozen noisy labels, and for the 21 datasets with ten or more classes it does not help at 40% asymmetric noise (53.8% versus 54.9%).

**The gain depends on the metric.** Accuracy rewards a classifier that drifts towards majority classes, and a large penalty shrinks the decision function towards the class-frequency intercept. Measured by balanced accuracy the wide grid changes performance under symmetric noise by −0.2, +0.4 and −0.1 points at 20%, 30% and 40%, and measured by macro-F1 by −0.6, −0.5 and −3.1 points (Holm-adjusted p = 0.028; 40 wins, 65 losses). The accuracy gain under symmetric noise is thus real only for roughly balanced problems: on the 59 datasets whose largest class is less than 1.5 times the smallest, balanced accuracy at 30% symmetric noise rises from 72.7% to 73.9%, whereas on the 22 datasets with a ratio of 3 or more accuracy rises from 70.1% to 73.8% while balanced accuracy does not move (61.8% and 61.6%). Under asymmetric and boundary noise the wide grid helps on every metric, by smaller amounts: +2.7 points (Holm-adjusted p < 0.001; 74 wins, 34 losses) in balanced accuracy and +1.6 points (Holm-adjusted p < 0.001; 59 wins, 29 losses) in macro-F1 at the heaviest rates. Proposition 1 concerns the arg-max of the expected decision function and is silent on which classes absorb the variance; the data say that it is the minority classes.

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

: **Table 2** Wide versus default penalty grid, all 108 datasets: paired differences in accuracy points, two-sided Wilcoxon signed-rank test, Holm correction over the twelve conditions, matched-pairs rank-biserial effect size, and wins/ties/losses over datasets


## 6.2 RQ2: cleaning detects mislabelled series but lowers accuracy

Every cleaning baseline is worse than wide-grid ridge on clean labels (Table 1): calibrated confident learning by 2.1 points (11 wins, 74 losses), 5-fold relabelling by 4.0, LOO relabelling by 4.6, LOO removal by 3.1 and edited nearest neighbours by 6.7 points. At 10% symmetric or asymmetric noise all of them remain below wide-grid ridge on average. Calibrated confident learning becomes useful at high symmetric noise, +1.5 points (95% CI +0.7 to +2.3; Holm-adjusted p = 0.001; 68 wins, 37 losses) at 30% and +1.4 points (95% CI +0.7 to +2.0; Holm-adjusted p < 0.001; 72 wins, 32 losses) at 40%, but not significantly so under asymmetric noise (+0.6 points (95% CI −0.4 to +1.6; Holm-adjusted p = 0.299; 65 wins, 43 losses) at 40%). Under boundary noise one-round LOO relabelling is the best method and gains 1.4 points at the 30% rate.

How the baseline is fed matters. Our first implementation passed raw ridge scores through a unit-temperature softmax, which yields nearly flat probabilities; fitting a single temperature improves confident learning at every symmetric and asymmetric rate, by 1.3 points on clean labels and 2.5 points at 30% symmetric noise (both Holm-adjusted p < 0.001). The uncalibrated variant would have understated the baseline, and all comparisons below use the calibrated one.

The reason is not poor detection. The AUROC of the LOO suspicion score is 0.94 at 10% and 0.91 at 20% symmetric noise, the same as for 5-fold scores (0.94 and 0.91) at a fraction of the cost (Fig. 5). But every clean series that a classifier cannot predict from the others is also flagged, and on a dataset with clean LOO error $e$ a disagreement rule removes or flips roughly a fraction $e$ of correct labels. A regularised ridge classifier tolerates a noisy label better than it tolerates the loss, or the inversion, of a hard but correct one.

![**Fig. 5** AUROC of label-error detection for each method's suspicion score](/home/user/rp_c_14/figures/fig_detection_auroc.png){width=6.3in}

## 6.3 RQ3: LOO-EM against a well-regularised ridge classifier

On the confirmation set (Table 3) LOO-EM changes clean accuracy by −0.3 points (95% CI −0.6 to −0.1; Holm-adjusted p = 0.051; 7 wins, 21 losses), and improves on wide-grid ridge by +1.3 points (95% CI +0.6 to +2.0; Holm-adjusted p = 0.026; 36 wins, 21 losses) at 30% symmetric noise and by +2.2 points (95% CI +1.1 to +3.5; Holm-adjusted p = 0.012; 28 wins, 10 losses) at 40% asymmetric noise. The remaining conditions show smaller mean gains (0.1–0.8 points for symmetric and asymmetric noise) that are not significant after Holm correction on 60 datasets, and no effect for boundary noise. Over all 108 datasets, which include the design data and are therefore optimistic for LOO-EM, the differences have the same signs and sizes: −0.3 points (95% CI −0.6 to −0.2; Holm-adjusted p = 0.003; 13 wins, 35 losses) on clean labels, +0.6 points (95% CI +0.1 to +1.1; Holm-adjusted p = 0.022; 64 wins, 35 losses) at 20% and +1.1 points (95% CI +0.6 to +1.6; Holm-adjusted p < 0.001; 55 wins, 26 losses) at 40% symmetric noise, and +2.0 points (95% CI +1.2 to +2.9; Holm-adjusted p < 0.001; 50 wins, 17 losses) at 40% asymmetric noise (Fig. 6). LOO-EM's gains do not depend on the metric: over all datasets it improves balanced accuracy over wide-grid ridge by +1.3 points (Holm-adjusted p < 0.001; 68 wins, 32 losses) at 30% symmetric and +1.9 points (Holm-adjusted p < 0.001; 54 wins, 16 losses) at 40% asymmetric noise, and macro-F1 by +1.2 and +1.8 points. Relative to the default pipeline the combined effect of the wide grid and LOO-EM on the confirmation set is +2.8 points (95% CI +1.5 to +4.0; Holm-adjusted p = 0.003; 42 wins, 16 losses) at 30% symmetric and +6.1 points (95% CI +3.2 to +8.9; Holm-adjusted p < 0.001; 47 wins, 12 losses) at 40% asymmetric noise.

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

: **Table 3** LOO-EM (BIC) versus wide-grid ridge on the 60 confirmation datasets (columns as in Table 2)


![**Fig. 6** Per-dataset accuracy of LOO-EM against wide-grid ridge; colour encodes training-set size](/home/user/rp_c_14/figures/fig_scatter_em_vs_ridge.png){width=6.3in}

Against calibrated confident learning, the strongest cleaning baseline, LOO-EM is better on clean labels (+2.0 points (95% CI +0.9 to +3.7; Holm-adjusted p < 0.001; 39 wins, 7 losses) on the confirmation set) and at 10% asymmetric noise (+1.5 points (95% CI +0.4 to +3.3; Holm-adjusted p = 0.017; 36 wins, 23 losses)), and statistically indistinguishable from it at noise rates of 20% and above (−0.5 points (95% CI −1.6 to +0.6; Holm-adjusted p = 1.000; 26 wins, 29 losses) at 40% symmetric and +1.6 points (95% CI +0.2 to +3.0; Holm-adjusted p = 0.353; 35 wins, 25 losses) at 40% asymmetric noise). What separates the two is therefore not accuracy under heavy noise but the price paid when the noise is light or absent, and the computational cost (Sect. 6.7).

LOO-EM also yields cleaner training labels: after correction 87.3% of labels are correct at 20% symmetric noise (80% before) and 87.0% at 20% asymmetric noise. The cost on clean data is real: BIC accepts a noise model in 28% of clean runs, and LOO-EM is then slightly worse than leaving the labels alone (13 wins, 35 losses over all datasets). The estimated noise rate is conservative (mean 0.15 at a true rate of 0.20; Fig. 7).

![**Fig. 7** Estimated against true noise rate for the full-matrix EM](/home/user/rp_c_14/figures/fig_noise_rate_estimation.png){width=6.3in}

## 6.4 Ablation, and the failure of the first version

Table 4 and Fig. 8 remove or alter one component at a time. *Leave-one-out scores are essential*: with in-sample fitted values the EM is indistinguishable from wide-grid ridge (63.7% versus 63.8% at 40% asymmetric noise) because BIC almost always prefers the noise-free model when the fit reproduces its targets. *The noise model is essential*: iterating on LOO probabilities without a transition matrix, i.e. LOO self-training, collapses to 73.0% on clean data. *Soft targets* matter little (hard targets: 65.2% versus 65.8% at 40% asymmetric noise). *The wide grid* matters under asymmetric and boundary noise (default grid: 64.2% and 75.8% versus 65.8% and 76.9%). *The symmetric model alone* matches the BIC choice except at 30–40% asymmetric noise, where the full matrix adds up to 1.8 points.

| Variant                  | S 0   | S 10   | S 20   | S 30   | S 40   | A 10   | A 20   | A 30   | A 40   | B 10   | B 20   | B 30   |
|:-------------------------|:------|:-------|:-------|:-------|:-------|:-------|:-------|:-------|:-------|:-------|:-------|:-------|
| LOO-EM (BIC, proposed)   | 86.8  | 84.1   | 80.2   | 74.8   | 66.8   | 84.7   | 81.9   | 74.1   | 65.8   | 83.1   | 80.0   | 76.9   |
| BIC: hard targets        | 86.9  | 84.2   | 80.1   | 74.4   | 66.4   | 84.8   | 81.9   | 74.2   | 65.2   | 83.2   | 79.9   | 76.9   |
| BIC: in-sample scores    | 87.2  | 83.9   | 79.6   | 73.9   | 65.8   | 84.5   | 81.3   | 73.2   | 63.7   | 83.2   | 80.0   | 76.8   |
| BIC: default grid        | 86.7  | 84.1   | 80.5   | 75.0   | 67.0   | 84.4   | 81.7   | 73.5   | 64.2   | 82.9   | 79.7   | 75.8   |
| BIC: 5 EM iterations     | 86.9  | 84.3   | 80.6   | 75.0   | 66.9   | 84.9   | 82.2   | 74.3   | 65.6   | 83.2   | 80.0   | 77.0   |
| BIC: penalty x0.5        | 86.7  | 84.1   | 80.4   | 74.9   | 67.1   | 84.6   | 81.9   | 74.5   | 66.9   | 83.1   | 80.1   | 76.8   |
| BIC: penalty x2          | 87.0  | 84.0   | 80.0   | 74.5   | 66.8   | 84.7   | 81.8   | 73.8   | 64.6   | 83.1   | 80.0   | 77.0   |
| Symmetric T only         | 86.9  | 84.1   | 80.2   | 74.8   | 67.0   | 84.8   | 82.0   | 73.8   | 64.0   | 83.1   | 80.0   | 77.0   |
| LOO-EM v1 (full T)       | 85.4  | 82.4   | 78.6   | 73.5   | 65.8   | 82.7   | 80.2   | 74.0   | 67.4   | 82.8   | 79.9   | 76.4   |
| v1 without gate          | 84.5  | 81.7   | 78.3   | 74.2   | 66.1   | 82.5   | 79.9   | 73.8   | 67.2   | 82.4   | 79.6   | 76.1   |
| LOO self-training (no T) | 73.0  | 67.6   | 61.6   | 55.0   | 48.4   | 67.8   | 62.4   | 56.4   | 48.4   | 73.3   | 72.4   | 68.3   |
| LOO relabel (1 round)    | 82.6  | 80.4   | 77.0   | 71.3   | 64.2   | 80.7   | 78.4   | 71.2   | 62.3   | 82.1   | 80.7   | 78.3   |
| LOO relabel (iterated)   | 79.6  | 77.3   | 74.2   | 69.6   | 62.7   | 77.5   | 75.5   | 69.8   | 60.7   | 79.2   | 77.9   | 75.5   |
| LOO remove               | 84.1  | 82.2   | 79.1   | 73.5   | 66.2   | 82.5   | 80.7   | 73.9   | 65.5   | 82.8   | 80.7   | 78.1   |
| Ridge (wide grid)        | 87.2  | 83.9   | 79.6   | 73.6   | 65.8   | 84.5   | 81.3   | 73.5   | 63.8   | 83.2   | 80.1   | 76.9   |

: **Table 4** Ablation and variants: mean test accuracy (%) over all datasets. S, A and B denote symmetric, asymmetric and boundary noise at the given rate (%)


![**Fig. 8** Accuracy change of each ablation relative to LOO-EM](/home/user/rp_c_14/figures/fig_ablation.png){width=6.3in}

The first version (v1) used the full matrix whenever a one-parameter likelihood gate accepted it. On the 12 development datasets, none of which has more than eight classes, it looked uniformly beneficial. On all 108 datasets it loses 1.8 points on clean labels (12 wins, 59 losses) and 1.5 points at 10% symmetric noise, and gains 3.6 points at 40% asymmetric noise, more than the BIC variant (2.0). The loss is concentrated where the matrix is over-parameterised: for the datasets with ten or more classes v1 scores 79.2% on clean data against 83.1% for ridge, yet at 40% asymmetric noise it is the only method that helps there (58.8% versus 54.0% for the BIC variant). This is the trade-off of Sect. 4.4 in numbers: under 40% asymmetric noise BIC accepts a noise model in only 36% of runs, because data in which 40% of each class is consistently relabelled as its neighbour are equally well explained as two overlapping classes. Halving the BIC penalty moves the method towards v1 (66.9% at 40% asymmetric noise, 86.7% on clean labels), doubling it moves it towards ridge (64.6% and 87.0%).

## 6.5 Robustness: training-set size, number of classes, noise model

Table 5 stratifies the main comparison. LOO-EM's gain over wide-grid ridge at 40% asymmetric noise grows with the training-set size: +0.8 points below 100 series, +2.1 between 100 and 500, and +3.8 for 500 or more. With fewer than 100 training series no method is reliably better than another under symmetric noise, and all methods lose more than 20 points at 30% noise. By number of classes, LOO-EM helps most for 3–9 classes (+4.4 points at 40% asymmetric noise), is neutral for two classes, where pair-flip noise is a single flip probability that the symmetric model already represents, and does not help with ten or more classes at that noise level.

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

: **Table 5** Mean test accuracy (%) by training-set size and by number of classes


Boundary noise is the hardest case for every label-based remedy. The mislabelled series are those the clean model was least sure about, relabelled to the class it considered second most likely; the widened grid recovers 2.2 points at the 30% rate, LOO-EM adds nothing (−0.0 points (95% CI −0.3 to +0.2; Holm-adjusted p = 1.000; 28 wins, 32 losses)), and simple LOO relabelling is the best method (78.3%).

## 6.6 RQ4: generalisation

Table 6 summarises three checks of external validity for the kernel-transform pipeline.

*Hydra features* (29 datasets, seeds 0–2). The datasets are the fixed random subset of the sensitivity study, without StarLightCurves, whose Hydra transform exceeded the available memory. Hydra counts are square-rooted and standardised here rather than passed through the sparse scaler of the original pipeline. The wide grid changes accuracy relative to the default grid by −0.8 points (Holm-adjusted p = 1.000; 6 wins, 4 losses) on clean labels, +2.5 points (Holm-adjusted p = 0.306; 18 wins, 10 losses) at 40% symmetric and +4.7 points (Holm-adjusted p = 0.026; 23 wins, 6 losses) at 40% asymmetric noise. LOO-EM relative to wide-grid ridge: +0.0 points (Holm-adjusted p = 1.000; 4 wins, 10 losses) on clean labels, +1.4 points (Holm-adjusted p = 0.165; 15 wins, 5 losses) at 40% symmetric and +0.5 points (Holm-adjusted p = 1.000; 9 wins, 4 losses) at 40% asymmetric noise.

*Multivariate series* (18 UEA datasets, seeds 0–2). Wide versus default grid: +2.4 points (Holm-adjusted p = 0.907; 13 wins, 5 losses) at 40% symmetric and +3.9 points (Holm-adjusted p = 0.182; 13 wins, 5 losses) at 40% asymmetric noise. LOO-EM versus wide-grid ridge: −0.9 points (Holm-adjusted p = 0.252; 1 wins, 6 losses) on clean labels and +1.4 points (Holm-adjusted p = 0.261; 8 wins, 4 losses) at 40% asymmetric noise. With this few datasets the tests have little power and the table should be read descriptively.

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

: **Table 6** Generalisation: mean test accuracy (%) with other random-kernel transforms, on multivariate UEA datasets and on the largest UCR datasets (seeds 0–2)


*Other classifier families* (Table 7; 108 datasets, seeds 0–2). QUANT, whose extremely randomised trees average over many weak learners, loses 19.7 points between clean labels and 40% symmetric noise (86.1% to 66.5%), compared with 22.8 points for default-grid MiniRocket and 21.6 for the wide grid. At 40% asymmetric noise QUANT reaches 61.8%, wide-grid ridge 63.8% and LOO-EM 65.9% (LOO-EM versus QUANT: +4.1 points (Holm-adjusted p < 0.001; 74 wins, 34 losses); at 40% symmetric noise −0.2 points (Holm-adjusted p = 1.000; 53 wins, 52 losses)). The 1-nearest-neighbour classifier is the most fragile method tested (71.9% clean, 48.7% at 40% symmetric noise).

*Deep baselines* (Table 8; 6 small datasets, seeds 0–2, CPU training). At 30% symmetric noise a fully convolutional network reaches 73.2% with cross-entropy, 68.3% with the generalised cross-entropy and 82.0% with Co-teaching given the true noise rate; wide-grid ridge reaches 78.1% and LOO-EM 78.1%. At 30% asymmetric noise the figures are 72.1%, 72.7%, 78.3%, 80.9% and 80.1%; on clean labels 91.1%, 93.6%, 92.1%, 94.4% and 94.4%. One network takes 17 s on average on one CPU core (Co-teaching 35 s). On these datasets Co-teaching, which is told the true noise rate, is the most accurate method at 30% symmetric noise, and the ridge variants are the most accurate on clean labels and under asymmetric noise. The comparison covers few, small datasets and simple networks and supports no general claim about deep noisy-label methods.

| Experiment       | Method               | sym 0%   | sym 20%   | sym 40%   | asym 20%   | asym 40%   | boundary 20%   |
|:-----------------|:---------------------|:---------|:----------|:----------|:-----------|:-----------|:---------------|
| 108 UCR datasets | Ridge (default grid) | 86.8     | 78.9      | 64.0      | 79.4       | 59.5       | 79.0           |
| 108 UCR datasets | Ridge (wide grid)    | 86.9     | 79.5      | 65.3      | 81.0       | 63.8       | 79.8           |
| 108 UCR datasets | LOO-EM (BIC)         | 86.5     | 80.0      | 66.3      | 81.5       | 65.9       | 80.0           |
| 108 UCR datasets | QUANT                | 86.1     | 80.0      | 66.5      | 80.5       | 61.8       | 78.6           |
| 108 UCR datasets | 1-NN Euclidean       | 71.9     | 60.8      | 48.7      | 60.2       | 48.2       | 67.2           |

: **Table 7** Other classifier families trained on the same noisy labels (seeds 0–2): mean test accuracy (%)


| Experiment           | Method                             | sym 0%   | sym 30%   | asym 30%   |
|:---------------------|:-----------------------------------|:---------|:----------|:-----------|
| 6 small UCR datasets | Ridge (default grid)               | 94.4     | 81.2      | 78.1       |
| 6 small UCR datasets | Ridge (wide grid)                  | 94.4     | 78.1      | 80.9       |
| 6 small UCR datasets | LOO-EM (BIC)                       | 94.4     | 78.1      | 80.1       |
| 6 small UCR datasets | FCN, cross-entropy                 | 91.1     | 73.2      | 72.1       |
| 6 small UCR datasets | FCN, GCE loss                      | 93.6     | 68.3      | 72.7       |
| 6 small UCR datasets | FCN, Co-teaching (true rate given) | 92.1     | 82.0      | 78.3       |

: **Table 8** Deep baselines on 6 small UCR datasets (seeds 0–2): mean test accuracy (%)


## 6.7 Computational cost

Table 9 and Fig. 9 report the time each method adds to the shared work, which is the MiniRocket transform (median 0.52 s per dataset) and one eigendecomposition (median 0.028 s). Selecting the penalty on the wide instead of the default grid costs 1 ms at the median. LOO-EM takes a median of 0.11 s (mean 0.28 s), 2.6 times the eigendecomposition; 5-fold confident learning takes 0.39 s (mean 1.04 s), 8.8 times. LOO-EM is therefore not free, as a single LOO relabelling round is (4 ms), but it is about 4 times cheaper on average than the cross-validated alternative and never refits a model. Memory is dominated by the $n\times n$ eigenvector matrix and its element-wise square. All ridge variants deploy the same linear classifier, so model size and inference cost are identical; energy use was not measured. On the largest dataset processed (Crop, 7,200 training series, two BLAS threads) the eigendecomposition takes 47 s and LOO-EM 27 s per noise condition.

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

: **Table 9** Wall-clock cost per run of each method beyond the shared feature transform and eigendecomposition (single CPU thread, all datasets and conditions)


![**Fig. 9** Wall-clock time of the shared eigendecomposition and of each cleaning method against training-set size (40% of datasets shown per method for the 30% asymmetric condition), single CPU thread](/home/user/rp_c_14/figures/fig_runtime.png){width=6.3in}

## 6.8 Sensitivity

Sensitivity was examined on a fixed random subset of 30 datasets outside the development set (seeds 0–2; full table in the supplement). Mean accuracy (%) on clean labels / at 40% symmetric / at 40% asymmetric noise is, for the reference setting, LOO-EM: 85.0 / 67.0 / 64.1; BIC penalty ×0.5: 84.8 / 66.9 / 65.2; BIC penalty ×2: 85.0 / 66.9 / 62.7; BIC penalty ×4: 85.1 / 66.7 / 62.7; prior strength 1: 84.9 / 66.6 / 64.1; prior strength 50: 84.7 / 66.6 / 62.9; 5 EM iterations: 85.0 / 66.9 / 64.2; 50 EM iterations: 85.0 / 67.1 / 64.1; wide-grid ridge: 85.2 / 65.9 / 62.2. The BIC penalty acts as described in Sect. 6.4: a smaller penalty buys accuracy under asymmetric noise with accuracy on clean labels. A strong prior on the transition matrix (50 pseudo-counts per class) pulls the full model towards the symmetric one and lowers accuracy under asymmetric noise; the number of EM iterations is immaterial within the range tried. Varying the number of kernels at 40% symmetric noise gives 1000 kernels: ridge 64.7, LOO-EM 65.5; 2500 kernels: ridge 65.5, LOO-EM 66.2; 5000 kernels: ridge 65.9, LOO-EM 66.9; 10,000 kernels: ridge 65.9, LOO-EM 67.0; the ordering of the methods does not depend on it.

## 6.9 Error analysis

Table 10 lists the datasets with the largest losses and gains of LOO-EM relative to wide-grid ridge, and Fig. 10 plots the gain against training-set size. On clean labels the largest losses occur on Beef, PigCVP, EthanolLevel, ScreenType and Worms: small or many-class problems on which the clean classifier itself is far from perfect, so that systematic confusions between similar classes are indistinguishable from label noise. LOO-EM changes the predictions of 44% of datasets on clean labels at all. At 40% asymmetric noise the largest gains (15–19 points on TwoPatterns, LargeKitchenAppliances, SmoothSubspace, UWaveGestureLibraryAll and OSULeaf) occur on problems with 3–8 classes and at least 150 training series, where the classes are separable enough for the full transition matrix to be identified; the largest losses (1–3 points) occur on two-class problems or with fewer than 60 training series. The per-dataset gain at 40% asymmetric noise correlates weakly with the training-set size (Spearman 0.20) and with clean accuracy (−0.15).

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

: **Table 10** Datasets with the largest losses and gains of LOO-EM (BIC) relative to wide-grid ridge (accuracy %, mean of five resamples)


![**Fig. 10** Gain of LOO-EM over wide-grid ridge against training-set size at 40% asymmetric noise and on clean labels](/home/user/rp_c_14/figures/fig_gain_vs_train_size.png){width=6.3in}

# 7 Discussion

**Practical recommendations.** (1) When labels may be noisy, widen the ridge penalty grid of random-kernel classifiers. It costs nothing, does not change clean accuracy, and gave the largest accuracy improvement in this study; on imbalanced problems under symmetric noise, check a class-balanced metric, because there the improvement in accuracy was bought with minority-class recall. (2) Do not remove or hard-relabel training series on the strength of out-of-sample disagreement alone; on these features that lowered accuracy on clean data and at 10% noise for every cleaning method, and confident learning repaid its cost only from about 30% symmetric noise. (3) If substantial class-conditional noise is plausible, and in particular systematic confusion between specific classes, LOO-EM with BIC adds a further gain at the cost of a few tenths of a point when the labels are in fact clean. (4) If heavy asymmetric noise is known to be present, forcing the full transition matrix (the v1 behaviour) is worth considering; without that knowledge it is not safe.

**Why regularisation goes so far.** Proposition 1 implies that under symmetric noise the ridge classifier needs no correction in expectation, only more averaging. The interpolating regime created by ten thousand features and a few hundred series is exactly where a narrow penalty grid prevents that averaging. That the same change helps under asymmetric and boundary noise shows that a large part of the damage in those settings is also variance.

**Relation to prior work.** The latent-label EM is classical (Dawid and Skene 1979; Bootkrajang and Kabán 2012); what is specific here is that the predictive distribution is the closed-form LOO one, which removes self-confirmation in the $p\gg n$ regime (the in-sample ablation is inert) and makes the iteration cheap, and that the noise model is selected rather than assumed. Prevalidated ridge regression (Dempster et al. 2024b) uses the same LOO identity for calibration on clean labels. We did not rerun the deep noisy-label methods of Castellani et al. (2021), Ma et al. (2023) or Liu et al. (2023); they require GPU-scale training budgets, and their published evaluations use different dataset subsets and splits, so we make no claim relative to them.

**Is the improvement worth its cost?** The widened grid is free. LOO-EM adds a small multiple of the eigendecomposition time that every LOO-tuned ridge classifier already spends and is cheaper than 5-fold confident learning. Its accuracy benefit over wide-grid ridge is modest on average and concentrated in specific regimes; we regard the regularisation finding, with its metric caveat, as the result with the widest practical consequence, and LOO-EM as the more uniform but smaller improvement.

# 8 Limitations

*Synthetic noise.* All noise is injected. We know of no public TSC benchmark with paired noisy and verified labels; the boundary model is our attempt at a realistic, feature-dependent case, and it is the one on which label-based correction failed. *One archive.* The UCR and UEA archives contain many small, pre-segmented problems, some with train–test duplicates and with series cut from the same recordings; absolute accuracies are archive-specific. *Design history.* The proposed variant was designed after an earlier one failed; the confirmation set guards against the resulting optimism, but it contains only 60 datasets, and several individual comparisons on it are not significant. *Baselines.* Deep noisy-label methods were not rerun; the deep baselines we did run are simple, small in number and trained on six small datasets. MultiRocket features were planned but not run. *Scale.* The $O(n^3)$ eigendecomposition limits the approach to roughly 10^4^ training series on a workstation; larger problems would need a primal or iterative formulation, for which the LOO identity is still available but the cached structure differs. *Class-conditional model.* LOO-EM assumes noise that depends only on the class; it has no mechanism for instance-dependent noise. *Identifiability.* Without outside knowledge, heavy asymmetric noise cannot be told from overlapping classes, and no choice of penalty in the BIC removes that.

# 9 Conclusion

For ridge classifiers on random convolutional kernel features, the first response to suspected label noise should be regularisation, not cleaning. Symmetric noise leaves the expected decision function order-preserving and inflates its variance, the default penalty grid is too narrow to counter that, and widening it recovers up to 4.2 accuracy points over 108 datasets at no cost on clean labels, although under symmetric noise the gain does not extend to class-balanced metrics. Generic cleaning pipelines detect mislabelled series accurately but reduce accuracy on clean and lightly corrupted data when their detections are acted upon. An EM procedure built on closed-form leave-one-out predictions with BIC model selection is the only correction we tested that improves on a well-regularised ridge classifier under moderate to heavy class-conditional noise while staying within a few tenths of a point on clean data (calibrated confident learning matches it under heavy noise but loses about two points on clean labels), and its gains hold on accuracy, balanced accuracy and macro-F1 alike; and its limits are set by an identifiability problem rather than by computation. Future work should test these conclusions on data with verified real-world label errors and extend the leave-one-out machinery to instance-dependent noise.

# Declarations

**Funding** [To be completed by the authors.]

**Competing interests** [To be completed by the authors; e.g. "The authors have no competing interests to declare that are relevant to the content of this article."]

**Ethics approval and consent** Not applicable. The study uses only public benchmark datasets and involves no human participants or animals.

**Data availability** All datasets are publicly available from the UCR/UEA time series classification archive (https://www.timeseriesclassification.com) and are downloaded automatically by the accompanying code.

**Code availability** The complete source code, configuration, raw per-run results, and the scripts that generate every table and figure in this article are available at [repository URL to be inserted by the authors].

**Author contributions** [To be completed by the authors.]

**Use of AI tools** [To be completed by the authors in line with the publisher's policy on the disclosure of AI assistance in the preparation of the code, experiments and manuscript.]

# References

Angluin D, Laird P (1988) Learning from noisy examples. Mach Learn 2:343–370. https://doi.org/10.1023/A:1022873112823

Atkinson G, Metsis V (2020) Identifying label noise in time-series datasets. In: Adjunct proceedings of the 2020 ACM international joint conference on pervasive and ubiquitous computing (UbiComp-ISWC '20). ACM, pp 238–243. https://doi.org/10.1145/3410530.3414366

Bagnall A, Lines J, Bostrom A, Large J, Keogh E (2017) The great time series classification bake off: a review and experimental evaluation of recent algorithmic advances. Data Min Knowl Discov 31:606–660. https://doi.org/10.1007/s10618-016-0483-9

Bartlett PL, Long PM, Lugosi G, Tsigler A (2020) Benign overfitting in linear regression. Proc Natl Acad Sci USA 117:30063–30070. https://doi.org/10.1073/pnas.1907378117

Bootkrajang J, Kabán A (2012) Label-noise robust logistic regression and its applications. In: Machine learning and knowledge discovery in databases (ECML PKDD 2012). Lecture notes in computer science, vol 7523. Springer, pp 143–158. https://doi.org/10.1007/978-3-642-33460-3_15

Brodley CE, Friedl MA (1999) Identifying mislabeled training data. J Artif Intell Res 11:131–167. https://doi.org/10.1613/jair.606

Castellani A, Schmitt S, Hammer B (2021) Estimating the electrical power output of industrial devices with end-to-end time-series classification in the presence of label noise. In: Machine learning and knowledge discovery in databases. Research track (ECML PKDD 2021). Lecture notes in computer science, vol 12975. Springer, pp 469–484. https://doi.org/10.1007/978-3-030-86486-6_29

Chen P, Ye J, Chen G, Zhao J, Heng PA (2021) Robustness of accuracy metric and its inspirations in learning with noisy labels. Proc AAAI Conf Artif Intell 35(13):11451–11461. https://doi.org/10.1609/aaai.v35i13.17364

Dau HA, Bagnall A, Kamgar K, Yeh CCM, Zhu Y, Gharghabi S, Ratanamahatana CA, Keogh E (2019) The UCR time series archive. IEEE/CAA J Autom Sin 6:1293–1305. https://doi.org/10.1109/JAS.2019.1911747

Dawid AP, Skene AM (1979) Maximum likelihood estimation of observer error-rates using the EM algorithm. J R Stat Soc Ser C Appl Stat 28:20–28. https://doi.org/10.2307/2346806

Dempster A, Petitjean F, Webb GI (2020) ROCKET: exceptionally fast and accurate time series classification using random convolutional kernels. Data Min Knowl Discov 34:1454–1495. https://doi.org/10.1007/s10618-020-00701-z

Dempster A, Schmidt DF, Webb GI (2021) MiniRocket: a very fast (almost) deterministic transform for time series classification. In: Proceedings of the 27th ACM SIGKDD conference on knowledge discovery and data mining. ACM, pp 248–257. https://doi.org/10.1145/3447548.3467231

Dempster A, Schmidt DF, Webb GI (2023) Hydra: competing convolutional kernels for fast and accurate time series classification. Data Min Knowl Discov 37:1779–1805. https://doi.org/10.1007/s10618-023-00939-3

Dempster A, Schmidt DF, Webb GI (2024a) quant: a minimalist interval method for time series classification. Data Min Knowl Discov 38:2377–2402. https://doi.org/10.1007/s10618-024-01036-9

Dempster A, Webb GI, Schmidt DF (2024b) Prevalidated ridge regression is a highly-efficient drop-in replacement for logistic regression for high-dimensional data. arXiv preprint arXiv:2401.15610

Demšar J (2006) Statistical comparisons of classifiers over multiple data sets. J Mach Learn Res 7:1–30

Frénay B, Verleysen M (2014) Classification in the presence of label noise: a survey. IEEE Trans Neural Netw Learn Syst 25:845–869. https://doi.org/10.1109/TNNLS.2013.2292894

Ghosh A, Kumar H, Sastry PS (2017) Robust loss functions under label noise for deep neural networks. Proc AAAI Conf Artif Intell 31(1). https://doi.org/10.1609/aaai.v31i1.10894

Golub GH, Heath M, Wahba G (1979) Generalized cross-validation as a method for choosing a good ridge parameter. Technometrics 21:215–223. https://doi.org/10.1080/00401706.1979.10489751

Han B, Yao Q, Yu X, Niu G, Xu M, Hu W, Tsang I, Sugiyama M (2018) Co-teaching: robust training of deep neural networks with extremely noisy labels. In: Advances in neural information processing systems 31 (NeurIPS 2018). arXiv:1804.06872

Hastie T, Tibshirani R, Friedman J (2009) The elements of statistical learning: data mining, inference, and prediction, 2nd edn. Springer, New York. https://doi.org/10.1007/978-0-387-84858-7

Hoerl AE, Kennard RW (1970) Ridge regression: biased estimation for nonorthogonal problems. Technometrics 12:55–67. https://doi.org/10.1080/00401706.1970.10488634

Holm S (1979) A simple sequentially rejective multiple test procedure. Scand J Stat 6:65–70

Ismail Fawaz H, Forestier G, Weber J, Idoumghar L, Muller PA (2019) Deep learning for time series classification: a review. Data Min Knowl Discov 33:917–963. https://doi.org/10.1007/s10618-019-00619-1

Ismail Fawaz H, Lucas B, Forestier G, Pelletier C, Schmidt DF, Weber J, Webb GI, Idoumghar L, Muller PA, Petitjean F (2020) InceptionTime: finding AlexNet for time series classification. Data Min Knowl Discov 34:1936–1962. https://doi.org/10.1007/s10618-020-00710-y

Li Z, Liu Y, Tanaka G (2026) Robust and efficient noisy-label time-series classification via dynamic time warping based granular ball computing. arXiv preprint arXiv:2608.11704

Liu T, Tao D (2016) Classification with noisy labels by importance reweighting. IEEE Trans Pattern Anal Mach Intell 38:447–461. https://doi.org/10.1109/TPAMI.2015.2456899

Liu Z, Ma P, Chen D, Pei W, Ma Q (2023) Scale-teaching: robust multi-scale training for time series classification with noisy labels. In: Advances in neural information processing systems 36 (NeurIPS 2023)

Ma P, Liu Z, Zheng J, Wang L, Ma Q (2023) CTW: confident time-warping for time-series label-noise learning. In: Proceedings of the thirty-second international joint conference on artificial intelligence (IJCAI-23), pp 4046–4054. https://doi.org/10.24963/ijcai.2023/450

Middlehurst M, Ismail-Fawaz A, Guillaume A, Holder C, Guijo-Rubio D, Bulatova G, Tsaprounis L, Mentel L, Walter M, Schäfer P, Bagnall A (2024b) aeon: a Python toolkit for learning from time series. arXiv preprint arXiv:2406.14231

Middlehurst M, Schäfer P, Bagnall A (2024a) Bake off redux: a review and experimental evaluation of recent time series classification algorithms. Data Min Knowl Discov 38:1958–2031. https://doi.org/10.1007/s10618-024-01022-1

Natarajan N, Dhillon IS, Ravikumar P, Tewari A (2013) Learning with noisy labels. In: Advances in neural information processing systems 26 (NIPS 2013)

Northcutt C, Jiang L, Chuang I (2021) Confident learning: estimating uncertainty in dataset labels. J Artif Intell Res 70:1373–1411. https://doi.org/10.1613/jair.1.12125

Patrini G, Rozza A, Menon AK, Nock R, Qu L (2017) Making deep neural networks robust to label noise: a loss correction approach. In: 2017 IEEE conference on computer vision and pattern recognition (CVPR), pp 2233–2241. https://doi.org/10.1109/CVPR.2017.240

Schwarz G (1978) Estimating the dimension of a model. Ann Stat 6:461–464. https://doi.org/10.1214/aos/1176344136

Song H, Kim M, Park D, Shin Y, Lee JG (2023) Learning from noisy labels with deep neural networks: a survey. IEEE Trans Neural Netw Learn Syst 34:8135–8153. https://doi.org/10.1109/TNNLS.2022.3152527

Tan CW, Dempster A, Bergmeir C, Webb GI (2022) MultiRocket: multiple pooling operators and transformations for fast and effective time series classification. Data Min Knowl Discov 36:1623–1646. https://doi.org/10.1007/s10618-022-00844-1

Tibshirani RJ, Efron B (2002) Pre-validation and inference in microarrays. Stat Appl Genet Mol Biol 1(1):Article 1. https://doi.org/10.2202/1544-6115.1000

Wang Z, Yan W, Oates T (2017) Time series classification from scratch with deep neural networks: a strong baseline. In: 2017 international joint conference on neural networks (IJCNN), pp 1578–1585. https://doi.org/10.1109/IJCNN.2017.7966039

Wilcoxon F (1945) Individual comparisons by ranking methods. Biometrics Bull 1:80–83. https://doi.org/10.2307/3001968

Wilson DL (1972) Asymptotic properties of nearest neighbor rules using edited data. IEEE Trans Syst Man Cybern SMC-2:408–421. https://doi.org/10.1109/TSMC.1972.4309137

Zhang Z, Sabuncu MR (2018) Generalized cross entropy loss for training deep neural networks with noisy labels. In: Advances in neural information processing systems 31 (NeurIPS 2018). arXiv:1805.07836
