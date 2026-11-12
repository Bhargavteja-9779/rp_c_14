---
title: "Regularise before you relabel: label noise in random convolutional kernel time series classification"
author: "[Author One]^1\\*^, [Author Two]^2^"
---

^1^ [Department, Institution, City, Country] · ^2^ [Department, Institution, City, Country]

\* Corresponding author: [name], [e-mail address], ORCID [0000-0000-0000-0000]

## Abstract

Random convolutional kernel transforms followed by a ridge classifier (ROCKET, MiniRocket and successors) are fast and accurate time series classifiers, but their behaviour under wrong training labels has not been studied systematically. We evaluate MiniRocket with ridge regression on <<N_DS>> UCR datasets, five resamples each, under symmetric, asymmetric and instance-dependent label noise of up to 40%. First, part of the fragility is an artefact of the customary penalty grid: at 30% symmetric noise the selected penalty sits on the grid's upper end in <<PCT_SAT('sym 30%')>>% of runs, and widening the grid raises mean accuracy by up to <<DMAX('ridge_wide','ridge_std')>> points with no loss on clean labels. We prove that symmetric noise leaves the expected ridge decision function order-preserving and only inflates its variance. The gain is metric-dependent: under symmetric noise it does not extend to balanced accuracy or macro-F1. Second, cleaning pipelines on the same features (confident learning, edited nearest neighbours, hard relabelling) detect mislabelled series well yet lower clean-data accuracy by <<-DPTS('conf_learning_cal','ridge_wide','sym 0%')>>–<<-DPTS('enn','ridge_wide','sym 0%')>> points; calibrated confident learning pays off only at 30–40% symmetric noise. Third, we propose LOO-EM, an expectation–maximisation procedure for class-conditional noise that always predicts with closed-form leave-one-out scores, reuses a single eigendecomposition and selects the noise model by BIC. On <<N_CONFIRM>> confirmation datasets it improves on wide-grid ridge by <<DPTS('loo_em_bic','ridge_wide','sym 30%','confirm')>> points at 30% symmetric and <<DPTS('loo_em_bic','ridge_wide','asym 40%','confirm')>> points at 40% asymmetric noise, costs <<-DPTS('loo_em_bic','ridge_wide','sym 0%','confirm')>> points on clean labels, and gains nothing under instance-dependent noise. An earlier variant that failed on many-class data is reported.

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

1. *A large-scale characterisation.* MiniRocket with ridge regression is evaluated on <<N_DS>> UCR datasets with five resamples and twelve noise conditions (<<N_ROWS>> recorded runs for <<N_METHODS>> method variants), with secondary experiments on Hydra and MultiRocket features, <<N_UEA>> multivariate UEA datasets, the four largest UCR datasets, and non-kernel and deep baselines.
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

$$f(x)=\bar y+k(x)^\top (K+\alpha I)^{-1}(Y-\mathbf{1}\bar y^\top),\qquad K=ZZ^\top,\quad k(x)=Z\,z(x), \tag{1}$$

where $\bar y$ is the vector of column means of $Y$. With the eigendecomposition $K=U\,\mathrm{diag}(s)\,U^\top$ the fitted training values are $HY$ with the hat matrix

$$H(\alpha)=\tfrac1n\mathbf{1}\mathbf{1}^\top+U\,\mathrm{diag}\!\left(\frac{s_j}{s_j+\alpha}\right)U^\top . \tag{2}$$

The prediction for sample $i$ of the model trained without sample $i$ is available without refitting (Golub et al. 1979; Hastie et al. 2009):

$$F^{\mathrm{loo}}_{i\cdot}=\frac{(HY)_{i\cdot}-h_{ii}Y_{i\cdot}}{1-h_{ii}},\qquad\text{i.e.}\quad F^{\mathrm{loo}}=M(\alpha)\,Y,\quad M=D^{-1}\big(H-\mathrm{diag}(h)\big),\ D=\mathrm{diag}(1-h_{ii}). \tag{3}$$

Two properties of Eq. (3) drive the rest of the paper. (a) $M(\alpha)$ has a zero diagonal: the LOO score of a sample does not depend on its own label. (b) $M(\alpha)$ does not depend on $Y$ at all: after any edit of the labels the complete set of LOO scores is recomputed by one $n\times n$ by $n\times C$ product. As in standard pipelines the column scaling is computed once on all training series and is not recomputed inside the LOO formula.

## 3.2 Symmetric noise adds variance, asymmetric noise adds bias

Suppose the observed labels $\tilde y_i$ are drawn from the true labels by a transition matrix $T$ with $T_{zk}=P(\tilde y=k\mid y=z)$, independently of $x$ given $y$. Write $\tilde Y$ for the encoded noisy labels and $g(x)=(f(x)+\mathbf 1)/2$ for the clean decision function on the 0/1 scale; because of the intercept the entries of $g(x)$ sum to one.

**Proposition 1** (symmetric noise). *Let $T=(1-\rho)I+\frac{\rho}{C-1}(\mathbf 1\mathbf 1^\top-I)$ with $\rho<(C-1)/C$. For every $x$ and every penalty $\alpha$, the decision function $\tilde f$ trained on $\tilde Y$ satisfies*

$$\mathbb{E}\big[\tilde f(x)\mid Y\big]=a\,f(x)+b\,\mathbf 1,\qquad a=1-\frac{\rho C}{C-1}>0,\quad b=\frac{\rho(2-C)}{C-1}, \tag{4}$$

*so the arg-max of the expected decision function equals that of the clean one. Moreover $\tilde f_c(x)-\mathbb{E}[\tilde f_c(x)\mid Y]=\sum_i w_i(x;\alpha)\,\varepsilon_{ic}$ with $w(x;\alpha)^\top=\tfrac1n\mathbf 1^\top+k(x)^\top(K+\alpha I)^{-1}(I-\tfrac1n\mathbf 1\mathbf 1^\top)$ and zero-mean $\varepsilon_{ic}$, independent across samples $i$, of variance at most 1; the non-constant part of $w$ shrinks monotonically to zero as $\alpha$ grows.*

*Proof.* Entry-wise, $\mathbb{E}[\tilde Y_{ic}\mid Y]$ equals $1-2\rho$ if $Y_{ic}=1$ and $-1+2\rho/(C-1)$ otherwise, i.e. $\mathbb{E}[\tilde Y\mid Y]=aY+b\mathbf 1\mathbf 1^\top$. Equation (1) is linear in the targets and reproduces constants through the intercept, which gives Eq. (4). The second statement is Eq. (1) applied to $\tilde Y-\mathbb{E}[\tilde Y\mid Y]$, whose rows are independent with bounded entries; the eigenvalues of $(K+\alpha I)^{-1}$ decrease in $\alpha$. $\square$

Symmetric noise therefore attenuates the signal by the factor $a$ and adds a variance term governed by $\|w(x;\alpha)\|$. The penalty that balances the two grows with $\rho$; a grid that is adequate for clean labels can become binding. This is the testable prediction behind RQ1.

**Proposition 2** (class-conditional noise in general). *For an arbitrary transition matrix, $\mathbb{E}[\tilde g(x)\mid Y]=T^\top g(x)$. The arg-max is preserved for all $g$ on the simplex if $T=aI+b\,\mathbf 1\mathbf 1^\top$ with $a>0$, which is the symmetric model; for other $T$ it need not be.*

*Proof.* On the 0/1 scale $\mathbb{E}[\tilde P\mid P]=PT$ and Eq. (1) is linear, so $\mathbb{E}[\tilde g(x)]^\top=g(x)^\top T$. If $T=aI+b\mathbf 1\mathbf 1^\top$, then $T^\top g=a\,g+b\,\mathbf 1$ because the entries of $g$ sum to one. For a counter-example with pair-flip noise ($T_{c,c}=1-\rho$, $T_{c,c+1}=\rho$, indices modulo $C$) take $C=3$, $\rho=0.3$ and $g=(0.45,0.40,0.15)$: then $T^\top g=(0.36,0.415,0.225)$ and the arg-max moves from the first to the second class. $\square$

Under asymmetric noise the expected decision function is thus wrong for points whose clean scores are not close to a vertex, however the penalty is chosen. Removing this bias requires a model of $T$, which motivates Sect. 4.

# 4 LOO-EM: expectation–maximisation on leave-one-out predictions

## 4.1 Latent-label model

We treat the true label $z_i$ as latent, with the usual class-conditional observation model $P(\tilde y_i=k\mid z_i=z,x_i)=T_{zk}$ (Dawid and Skene 1979; Bootkrajang and Kabán 2012). A classifier that is refitted on its own posteriors and then asked to score the same training points confirms itself; with $p\gg n$ a nearly interpolating ridge fit reproduces whatever targets it is given. We therefore let the predictive distribution for sample $i$ always be the leave-one-out one. Let $Q\in[0,1]^{n\times C}$ hold the current posteriors $q_i(z)$, initialised to the one-hot observed labels. One iteration consists of:

*M-step (classifier).* Fit the ridge model to the soft targets $2Q-1$ and obtain $F^{\mathrm{loo}}=M(\alpha)(2Q-1)$ by Eq. (3), with $\alpha$ re-selected by the LOO squared error on the current targets.

*M-step (noise model).* With $p_i(z)=\mathrm{softmax}(\beta F^{\mathrm{loo}}_{i\cdot})_z$, maximise the observed-label LOO log-likelihood

$$\ell(\beta,T)=\sum_{i=1}^{n}\log\sum_{z=1}^{C}p_i(z)\,T_{z\tilde y_i} \tag{5}$$

by alternating a one-dimensional search over the inverse temperature $\beta$ and the closed-form update $T_{zk}\propto N_{zk}+m\,T^{0}_{zk}$, where $N_{zk}=\sum_i r_i(z)\mathbb{1}[\tilde y_i=k]$, $r_i(z)\propto p_i(z)T_{z\tilde y_i}$, and $T^0$ is a symmetric matrix with the current overall noise rate (prior strength $m=C$). For the symmetric model only the rate is estimated.

*E-step.* $q_i(z)\propto p_i(z)\,T_{z\tilde y_i}$.

The loop stops when the largest change in $Q$ is below $10^{-3}$ or after 20 iterations. The final classifier is the ridge fit on the converged soft targets.

## 4.2 Choosing the noise model

A full transition matrix has $C(C-1)$ free parameters and can absorb ordinary classification error as "noise". We fit the symmetric model ($k=1$) and the full model ($k=C(C-1)$) and compare them with the noise-free model ($k=0$, $T=I$, $\beta$ fitted on the first-round LOO scores) by the Bayesian information criterion (Schwarz 1978) applied to Eq. (5), $\ell-\tfrac{k}{2}\log n$. If the noise-free model wins, the labels are left untouched and the method reduces to wide-grid ridge. Fig. 1 summarises the procedure.

![Overview of LOO-EM. The eigendecomposition of the centred Gram matrix is computed once; every EM iteration and the final fit reuse it](FIG:fig_method_overview)

## 4.3 Cost

The eigendecomposition costs $O(n^3)$ (or $O(np^2)$ through the primal when $n>p$) and is needed by any LOO-tuned ridge classifier. Each EM iteration costs $O(n^2C)$ per candidate penalty; the hat diagonal for each penalty is label-independent and cached, and after the first iteration only penalties adjacent to the current one are examined, extending the window while the optimum lies on its edge. No model is ever refitted from scratch. In contrast, k-fold cleaning needs $k$ further eigendecompositions.

## 4.4 What the likelihood cannot decide

Equation (5) cannot by itself distinguish heavy class-conditional noise from genuinely overlapping classes: a clean-posterior model with large off-diagonal $T$ and a flat predictor with $T=I$ can reach the same likelihood. Any procedure that corrects aggressively when predictions and labels disagree will therefore also "correct" hard but correctly labelled data. BIC resolves the ambiguity conservatively. Section 6.4 quantifies the price of that choice.

# 5 Experimental setup

**Data.** The 112 equal-length univariate problems of the UCR archive (Dau et al. 2019) were obtained through the aeon toolkit (Middlehurst et al. 2024b). The <<N_DS>> problems with fewer than 3,000 training series form the main benchmark (training sizes 16–1,800, median <<MED_N>>; 2–60 classes; Fig. 2); the four largest (FordA, FordB, Crop, ElectricDevices; 3,601–8,926 training series) are used for a separate scalability experiment because the 5-fold baselines are too expensive there. Generalisation to multivariate data uses the equal-length UEA problems that fit the available machine (<<N_UEA>> datasets).

![Training-set size, number of classes and series length of the UCR datasets used](FIG:fig_dataset_distribution)

**Resampling and seeds.** Five seeds (0–4) were fixed in advance. Seed 0 is the archive split; seeds 1–4 are stratified resamples of the pooled data with the original training and test sizes. Secondary experiments use seeds 0–2. Per-dataset scores are means over seeds.

**Noise.** Noise is applied to training labels only; test labels are always clean. *Symmetric*: a fraction $\rho\in\{0,0.1,0.2,0.3,0.4\}$ of labels is replaced by a uniformly chosen different class. *Asymmetric*: a fraction $\rho\in\{0.1,\dots,0.4\}$ of each class $c$ is relabelled $c+1$ (mod $C$). *Boundary*: the fraction $\rho\in\{0.1,0.2,0.3\}$ of training series with the smallest clean LOO margin is relabelled to the strongest competing class; this instance-dependent model is adversarial to margin-based detectors by construction. Each noise realisation is a deterministic function of dataset, seed, noise model and rate, so all methods see identical labels. Conditions in which a class disappears from the noisy training set are skipped for that dataset and seed.

**Features and classifier.** MiniRocket with 10,000 kernels as implemented in aeon, fitted on training series only; columns standardised with training statistics; ridge classifier as in Sect. 3.1. Our implementation reproduces scikit-learn's `RidgeClassifierCV` (identical selected penalty and predictions in a unit test). The *default grid* is ten log-spaced penalties in [10^−3^, 10^3^]; the *wide grid* is 25 log-spaced penalties in [10^−3^, 10^9^].

**Methods compared on identical features and noise.** (i) Ridge with the default grid; (ii) ridge with the wide grid; (iii) LOO-EM with BIC selection (proposed); (iv) LOO-EM v1, the earlier frozen variant with a full transition matrix accepted through a one-parameter gate; (v) confident learning (Northcutt et al. 2021) using the `cleanlab` implementation with 5-fold out-of-sample ridge scores passed through a softmax whose temperature is fitted by maximum likelihood on those scores, flagged series removed (the uncalibrated variant with unit temperature, which we ran first, is kept in the tables because the difference is instructive); (vi) relabelling from the same 5-fold predictions; (vii) edited nearest neighbours with $k=3$ in the standardised feature space (Wilson 1972); (viii) one-round relabelling and (ix) removal of series whose LOO prediction disagrees with their label. Methods (iii)–(ix) use the wide grid. Other classifier families trained on the same noisy labels: QUANT (Dempster et al. 2024a) and 1-nearest-neighbour with Euclidean distance, and, on eight small datasets, a fully convolutional network (Wang et al. 2017) trained with cross-entropy, with the generalised cross-entropy loss (Zhang and Sabuncu 2018) and with Co-teaching (Han et al. 2018) given the true noise rate. Deviations from original implementations: the deep baselines are trained for 100 epochs with Adam (learning rate 10^−3^, batch size up to 32) without early stopping, since no clean validation labels exist.

**Protocol and the two design stages.** Method design used 12 development datasets drawn at random. The resulting configuration (v1) was frozen and run on all datasets. An interim inspection of the first completed datasets showed that v1 lost accuracy on clean, many-class data (Sect. 6.4). We then designed the BIC variant on a *design set* of 39 datasets and froze it before any other result was inspected. The <<N_CONFIRM>> datasets that belong to neither the development nor the design set form the *confirmation set* on which the primary comparison is made. An anchor-point variant (Patrini et al. 2017) tried at the same stage was worse than wide-grid ridge on the design set and was dropped. The full history is in the supplementary protocol.

**Metrics and statistics.** The primary metric is test accuracy. Balanced accuracy and macro-F1 are reported alongside it where they change a conclusion (Sect. 6.1) and in full in the supplement. For label-error detection we report the AUROC of each method's suspicion score and the fraction of training labels that are correct after cleaning. Methods are compared by two-sided Wilcoxon signed-rank tests (Wilcoxon 1945) over datasets (unit of analysis: the dataset, as recommended by Demšar 2006), with Holm correction over the twelve noise conditions (Holm 1979), the matched-pairs rank-biserial correlation as effect size, and percentile bootstrap 95% confidence intervals (10,000 resamples of datasets) for the mean difference. The test is chosen because per-dataset accuracy differences are paired, bounded and not normally distributed.

**Leakage audit.** Transforms, scaling, penalty selection and all EM quantities use training data only. A duplicate scan of the archive splits found exact train–test duplicates in <<N_DUPDS>> datasets; <<N_DUP1>> datasets have more than 1% duplicated test series (<<DUPLIST>>). This affects all methods equally; the main comparison is repeated without these datasets in the supplement and is unchanged in direction.

**Implementation.** Python 3.13, aeon 1.6, scikit-learn 1.9, NumPy/SciPy, cleanlab 2.9, PyTorch 2.15 (CPU). All experiments ran on a four-core CPU machine with 15 GB of memory; no GPU was used.

# 6 Results

## 6.1 RQ1: fragility, and how much of it is the penalty grid

Table 1 and Fig. 3 give mean test accuracy over the <<N_DS>> datasets. With the default grid, accuracy falls from <<A('ridge_std','sym 0%')>>% on clean labels to <<A('ridge_std','sym 20%')>>% at 20% and <<A('ridge_std','sym 40%')>>% at 40% symmetric noise, and to <<A('ridge_std','asym 40%')>>% at 40% asymmetric noise.

TABLE:main

![Mean test accuracy over datasets as a function of the noise rate for the three noise models](FIG:fig_accuracy_vs_noise)

The selected penalty tells the story predicted by Proposition 1 (Fig. 4). On clean labels the default grid's upper end is selected in <<PCT_SAT('sym 0%')>>% of runs; at 20% symmetric noise in <<PCT_SAT('sym 20%')>>% and at 30% in <<PCT_SAT('sym 30%')>>%. With the wide grid, <<PCT_ABOVE('sym 30%')>>% of runs at 30% noise select a penalty above 10^3^.

![Distribution of the LOO-selected penalty under symmetric noise for the default and the wide grid](FIG:fig_selected_alpha)

Widening the grid leaves clean accuracy unchanged (<<DSTAT('ridge_wide','ridge_std','sym 0%')>>) and helps under every noise model (Table 2): <<DSTAT('ridge_wide','ridge_std','sym 30%')>> at 30% symmetric, <<DSTAT('ridge_wide','ridge_std','asym 40%')>> at 40% asymmetric and <<DSTAT('ridge_wide','ridge_std','boundary 30%')>> at 30% boundary noise. The gain under asymmetric noise is larger than under symmetric noise, although regularisation cannot remove the bias of Proposition 2: it removes the variance component, which is present under every noise model. The benefit is not uniform. For the <<NBIN('n<100')>> datasets with fewer than 100 training series the wide grid is worse than the default at 30% symmetric noise (<<BIN('n<100','sym 30%','ridge_wide')>>% versus <<BIN('n<100','sym 30%','ridge_std')>>%), where the LOO criterion itself is computed from a few dozen noisy labels, and for the <<NCB('C>=10')>> datasets with ten or more classes it does not help at 40% asymmetric noise (<<CB('C>=10','asym 40%','ridge_wide')>>% versus <<CB('C>=10','asym 40%','ridge_std')>>%).

**The gain depends on the metric.** Accuracy rewards a classifier that drifts towards majority classes, and a large penalty shrinks the decision function towards the class-frequency intercept. Measured by balanced accuracy the wide grid changes performance under symmetric noise by <<MD('bal_acc','ridge_wide','ridge_std','sym 20%')>>, <<MD('bal_acc','ridge_wide','ridge_std','sym 30%')>> and <<MD('bal_acc','ridge_wide','ridge_std','sym 40%')>> points at 20%, 30% and 40%, and measured by macro-F1 by <<MD('macro_f1','ridge_wide','ridge_std','sym 20%')>>, <<MD('macro_f1','ridge_wide','ridge_std','sym 30%')>> and <<MDP('macro_f1','ridge_wide','ridge_std','sym 40%')>>. The accuracy gain under symmetric noise is thus real only for roughly balanced problems: on the <<NIMB('balanced')>> datasets whose largest class is less than 1.5 times the smallest, balanced accuracy at 30% symmetric noise rises from <<IMB('bal_acc','balanced','sym 30%','ridge_std')>>% to <<IMB('bal_acc','balanced','sym 30%','ridge_wide')>>%, whereas on the <<NIMB('imbalanced')>> datasets with a ratio of 3 or more accuracy rises from <<IMB('acc','imbalanced','sym 30%','ridge_std')>>% to <<IMB('acc','imbalanced','sym 30%','ridge_wide')>>% while balanced accuracy does not move (<<IMB('bal_acc','imbalanced','sym 30%','ridge_std')>>% and <<IMB('bal_acc','imbalanced','sym 30%','ridge_wide')>>%). Under asymmetric and boundary noise the wide grid helps on every metric, by smaller amounts: <<MDP('bal_acc','ridge_wide','ridge_std','asym 40%')>> in balanced accuracy and <<MDP('macro_f1','ridge_wide','ridge_std','boundary 30%')>> in macro-F1 at the heaviest rates. Proposition 1 concerns the arg-max of the expected decision function and is silent on which classes absorb the variance; the data say that it is the minority classes.

TABLE:stats_wide

## 6.2 RQ2: cleaning detects mislabelled series but lowers accuracy

Every cleaning baseline is worse than wide-grid ridge on clean labels (Table 1): calibrated confident learning by <<-DPTS('conf_learning_cal','ridge_wide','sym 0%')>> points (<<WL('conf_learning_cal','ridge_wide','sym 0%')>>), 5-fold relabelling by <<-DPTS('kfold_relabel','ridge_wide','sym 0%')>>, LOO relabelling by <<-DPTS('loo_relabel','ridge_wide','sym 0%')>>, LOO removal by <<-DPTS('loo_remove','ridge_wide','sym 0%')>> and edited nearest neighbours by <<-DPTS('enn','ridge_wide','sym 0%')>> points. At 10% symmetric or asymmetric noise all of them remain below wide-grid ridge on average. Calibrated confident learning becomes useful at high symmetric noise, <<DSTAT('conf_learning_cal','ridge_wide','sym 30%')>> at 30% and <<DSTAT('conf_learning_cal','ridge_wide','sym 40%')>> at 40%, but not significantly so under asymmetric noise (<<DSTAT('conf_learning_cal','ridge_wide','asym 40%')>> at 40%). Under boundary noise one-round LOO relabelling is the best method and gains <<DPTS('loo_relabel','ridge_wide','boundary 30%')>> points at the 30% rate.

How the baseline is fed matters. Our first implementation passed raw ridge scores through a unit-temperature softmax, which yields nearly flat probabilities; fitting a single temperature improves confident learning at every symmetric and asymmetric rate, by <<DPTS('conf_learning_cal','conf_learning','sym 0%')>> points on clean labels and <<DPTS('conf_learning_cal','conf_learning','sym 30%')>> points at 30% symmetric noise (both Holm-adjusted p < 0.001). The uncalibrated variant would have understated the baseline, and all comparisons below use the calibrated one.

The reason is not poor detection. The AUROC of the LOO suspicion score is <<DET('loo_relabel','sym 10%')>> at 10% and <<DET('loo_relabel','sym 20%')>> at 20% symmetric noise, the same as for 5-fold scores (<<DET('conf_learning_cal','sym 10%')>> and <<DET('conf_learning_cal','sym 20%')>>) at a fraction of the cost (Fig. 5). But every clean series that a classifier cannot predict from the others is also flagged, and on a dataset with clean LOO error $e$ a disagreement rule removes or flips roughly a fraction $e$ of correct labels. A regularised ridge classifier tolerates a noisy label better than it tolerates the loss, or the inversion, of a hard but correct one.

![AUROC of label-error detection for each method's suspicion score](FIG:fig_detection_auroc)

## 6.3 RQ3: LOO-EM against a well-regularised ridge classifier

On the confirmation set (Table 3) LOO-EM changes clean accuracy by <<DSTAT('loo_em_bic','ridge_wide','sym 0%','confirm')>>, and improves on wide-grid ridge by <<DSTAT('loo_em_bic','ridge_wide','sym 30%','confirm')>> at 30% symmetric noise and by <<DSTAT('loo_em_bic','ridge_wide','asym 40%','confirm')>> at 40% asymmetric noise. The remaining conditions show smaller mean gains (0.1–0.8 points for symmetric and asymmetric noise) that are not significant after Holm correction on <<N_CONFIRM>> datasets, and no effect for boundary noise. Over all <<N_DS>> datasets, which include the design data and are therefore optimistic for LOO-EM, the differences have the same signs and sizes: <<DSTAT('loo_em_bic','ridge_wide','sym 0%')>> on clean labels, <<DSTAT('loo_em_bic','ridge_wide','sym 20%')>> at 20% and <<DSTAT('loo_em_bic','ridge_wide','sym 40%')>> at 40% symmetric noise, and <<DSTAT('loo_em_bic','ridge_wide','asym 40%')>> at 40% asymmetric noise (Fig. 6). LOO-EM's gains do not depend on the metric: over all datasets it improves balanced accuracy over wide-grid ridge by <<MDP('bal_acc','loo_em_bic','ridge_wide','sym 30%')>> at 30% symmetric and <<MDP('bal_acc','loo_em_bic','ridge_wide','asym 40%')>> at 40% asymmetric noise, and macro-F1 by <<MD('macro_f1','loo_em_bic','ridge_wide','sym 30%')>> and <<MD('macro_f1','loo_em_bic','ridge_wide','asym 40%')>> points. Relative to the default pipeline the combined effect of the wide grid and LOO-EM on the confirmation set is <<DSTAT('loo_em_bic','ridge_std','sym 30%','confirm')>> at 30% symmetric and <<DSTAT('loo_em_bic','ridge_std','asym 40%','confirm')>> at 40% asymmetric noise.

TABLE:stats_em

![Per-dataset accuracy of LOO-EM against wide-grid ridge; colour encodes training-set size](FIG:fig_scatter_em_vs_ridge)

Against calibrated confident learning, the strongest cleaning baseline, LOO-EM is better on clean labels (<<DSTAT('loo_em_bic','conf_learning_cal','sym 0%','confirm')>> on the confirmation set) and at 10% asymmetric noise (<<DSTAT('loo_em_bic','conf_learning_cal','asym 10%','confirm')>>), and statistically indistinguishable from it at noise rates of 20% and above (<<DSTAT('loo_em_bic','conf_learning_cal','sym 40%','confirm')>> at 40% symmetric and <<DSTAT('loo_em_bic','conf_learning_cal','asym 40%','confirm')>> at 40% asymmetric noise). What separates the two is therefore not accuracy under heavy noise but the price paid when the noise is light or absent, and the computational cost (Sect. 6.7).

LOO-EM also yields cleaner training labels: after correction <<LACC('loo_em_bic','sym 20%')>>% of labels are correct at 20% symmetric noise (80% before) and <<LACC('loo_em_bic','asym 20%')>>% at 20% asymmetric noise. The cost on clean data is real: BIC accepts a noise model in <<ACC_RATE('sym 0%')>>% of clean runs, and LOO-EM is then slightly worse than leaving the labels alone (<<WL('loo_em_bic','ridge_wide','sym 0%')>> over all datasets). The estimated noise rate is conservative (mean <<EST('sym 20%')>> at a true rate of 0.20; Fig. 7).

![Estimated against true noise rate for the full-matrix EM](FIG:fig_noise_rate_estimation)

## 6.4 Ablation, and the failure of the first version

Table 4 and Fig. 8 remove or alter one component at a time. *Leave-one-out scores are essential*: with in-sample fitted values the EM is indistinguishable from wide-grid ridge (<<A('bic_in_sample','asym 40%')>>% versus <<A('ridge_wide','asym 40%')>>% at 40% asymmetric noise) because BIC almost always prefers the noise-free model when the fit reproduces its targets. *The noise model is essential*: iterating on LOO probabilities without a transition matrix, i.e. LOO self-training, collapses to <<A('em_no_T','sym 0%')>>% on clean data. *Soft targets* matter little (hard targets: <<A('bic_hard','asym 40%')>>% versus <<A('loo_em_bic','asym 40%')>>% at 40% asymmetric noise). *The wide grid* matters under asymmetric and boundary noise (default grid: <<A('bic_std_grid','asym 40%')>>% and <<A('bic_std_grid','boundary 30%')>>% versus <<A('loo_em_bic','asym 40%')>>% and <<A('loo_em_bic','boundary 30%')>>%). *The symmetric model alone* matches the BIC choice except at 30–40% asymmetric noise, where the full matrix adds up to <<PTS('loo_em_bic','em_sym_T','asym 40%')>> points.

TABLE:ablation

![Accuracy change of each ablation relative to LOO-EM](FIG:fig_ablation)

The first version (v1) used the full matrix whenever a one-parameter likelihood gate accepted it. On the 12 development datasets, none of which has more than eight classes, it looked uniformly beneficial. On all <<N_DS>> datasets it loses <<-DPTS('loo_em','ridge_wide','sym 0%')>> points on clean labels (<<WL('loo_em','ridge_wide','sym 0%')>>) and <<-DPTS('loo_em','ridge_wide','sym 10%')>> points at 10% symmetric noise, and gains <<DPTS('loo_em','ridge_wide','asym 40%')>> points at 40% asymmetric noise, more than the BIC variant (<<DPTS('loo_em_bic','ridge_wide','asym 40%')>>). The loss is concentrated where the matrix is over-parameterised: for the datasets with ten or more classes v1 scores <<CB('C>=10','sym 0%','loo_em')>>% on clean data against <<CB('C>=10','sym 0%','ridge_wide')>>% for ridge, yet at 40% asymmetric noise it is the only method that helps there (<<CB('C>=10','asym 40%','loo_em')>>% versus <<CB('C>=10','asym 40%','loo_em_bic')>>% for the BIC variant). This is the trade-off of Sect. 4.4 in numbers: under 40% asymmetric noise BIC accepts a noise model in only <<ACC_RATE('asym 40%')>>% of runs, because data in which 40% of each class is consistently relabelled as its neighbour are equally well explained as two overlapping classes. Halving the BIC penalty moves the method towards v1 (<<A('bic_pen_0.5','asym 40%')>>% at 40% asymmetric noise, <<A('bic_pen_0.5','sym 0%')>>% on clean labels), doubling it moves it towards ridge (<<A('bic_pen_2','asym 40%')>>% and <<A('bic_pen_2','sym 0%')>>%).

## 6.5 Robustness: training-set size, number of classes, noise model

Table 5 stratifies the main comparison. LOO-EM's gain over wide-grid ridge at 40% asymmetric noise grows with the training-set size: <<BINDIFF('n<100','asym 40%')>> points below 100 series, <<BINDIFF('100<=n<500','asym 40%')>> between 100 and 500, and <<BINDIFF('n>=500','asym 40%')>> for 500 or more. With fewer than 100 training series no method is reliably better than another under symmetric noise, and all methods lose more than 20 points at 30% noise. By number of classes, LOO-EM helps most for 3–9 classes (<<CBDIFF('3<=C<10','asym 40%')>> points at 40% asymmetric noise), is neutral for two classes, where pair-flip noise is a single flip probability that the symmetric model already represents, and does not help with ten or more classes at that noise level.

TABLE:strata

Boundary noise is the hardest case for every label-based remedy. The mislabelled series are those the clean model was least sure about, relabelled to the class it considered second most likely; the widened grid recovers <<DPTS('ridge_wide','ridge_std','boundary 30%')>> points at the 30% rate, LOO-EM adds nothing (<<DSTAT('loo_em_bic','ridge_wide','boundary 30%')>>), and simple LOO relabelling is the best method (<<A('loo_relabel','boundary 30%')>>%).

## 6.6 RQ4: generalisation

SECTION:generalisation

## 6.7 Computational cost

SECTION:cost

![Wall-clock time of the shared eigendecomposition and of each cleaning method against training-set size (40% of datasets shown per method for the 30% asymmetric condition), single CPU thread](FIG:fig_runtime)

## 6.8 Sensitivity

SECTION:sensitivity

## 6.9 Error analysis

TABREF:errors lists the datasets with the largest losses and gains of LOO-EM relative to wide-grid ridge, and Fig. 10 plots the gain against training-set size. On clean labels the largest losses occur on Beef, PigCVP, EthanolLevel, ScreenType and Worms: small or many-class problems on which the clean classifier itself is far from perfect, so that systematic confusions between similar classes are indistinguishable from label noise. LOO-EM changes the predictions of <<FRAC_CHANGED('sym 0%')>>% of datasets on clean labels at all. At 40% asymmetric noise the largest gains (15–19 points on TwoPatterns, LargeKitchenAppliances, SmoothSubspace, UWaveGestureLibraryAll and OSULeaf) occur on problems with 3–8 classes and at least 150 training series, where the classes are separable enough for the full transition matrix to be identified; the largest losses (1–3 points) occur on two-class problems or with fewer than 60 training series. The per-dataset gain at 40% asymmetric noise correlates weakly with the training-set size (Spearman <<CORR('asym 40%','spearman_diff_vs_log_n')>>) and with clean accuracy (<<CORR('asym 40%','spearman_diff_vs_clean_accuracy')>>).

TABLE:errors

![Gain of LOO-EM over wide-grid ridge against training-set size at 40% asymmetric noise and on clean labels](FIG:fig_gain_vs_train_size)

# 7 Discussion

**Practical recommendations.** (1) When labels may be noisy, widen the ridge penalty grid of random-kernel classifiers. It costs nothing, does not change clean accuracy, and gave the largest accuracy improvement in this study; on imbalanced problems under symmetric noise, check a class-balanced metric, because there the improvement in accuracy was bought with minority-class recall. (2) Do not remove or hard-relabel training series on the strength of out-of-sample disagreement alone; on these features that lowered accuracy on clean data and at 10% noise for every cleaning method, and confident learning repaid its cost only from about 30% symmetric noise. (3) If substantial class-conditional noise is plausible, and in particular systematic confusion between specific classes, LOO-EM with BIC adds a further gain at the cost of a few tenths of a point when the labels are in fact clean. (4) If heavy asymmetric noise is known to be present, forcing the full transition matrix (the v1 behaviour) is worth considering; without that knowledge it is not safe.

**Why regularisation goes so far.** Proposition 1 implies that under symmetric noise the ridge classifier needs no correction in expectation, only more averaging. The interpolating regime created by ten thousand features and a few hundred series is exactly where a narrow penalty grid prevents that averaging. That the same change helps under asymmetric and boundary noise shows that a large part of the damage in those settings is also variance.

**Relation to prior work.** The latent-label EM is classical (Dawid and Skene 1979; Bootkrajang and Kabán 2012); what is specific here is that the predictive distribution is the closed-form LOO one, which removes self-confirmation in the $p\gg n$ regime (the in-sample ablation is inert) and makes the iteration cheap, and that the noise model is selected rather than assumed. Prevalidated ridge regression (Dempster et al. 2024b) uses the same LOO identity for calibration on clean labels. We did not rerun the deep noisy-label methods of Castellani et al. (2021), Ma et al. (2023) or Liu et al. (2023); they require GPU-scale training budgets, and their published evaluations use different dataset subsets and splits, so we make no claim relative to them.

**Is the improvement worth its cost?** The widened grid is free. LOO-EM adds a small multiple of the eigendecomposition time that every LOO-tuned ridge classifier already spends and is cheaper than 5-fold confident learning. Its accuracy benefit over wide-grid ridge is modest on average and concentrated in specific regimes; we regard the regularisation finding, with its metric caveat, as the result with the widest practical consequence, and LOO-EM as the more uniform but smaller improvement.

# 8 Limitations

*Synthetic noise.* All noise is injected. We know of no public TSC benchmark with paired noisy and verified labels; the boundary model is our attempt at a realistic, feature-dependent case, and it is the one on which label-based correction failed. *One archive.* The UCR and UEA archives contain many small, pre-segmented problems, some with train–test duplicates and with series cut from the same recordings; absolute accuracies are archive-specific. *Design history.* The proposed variant was designed after an earlier one failed; the confirmation set guards against the resulting optimism, but it contains only <<N_CONFIRM>> datasets, and several individual comparisons on it are not significant. *Baselines.* Deep noisy-label methods were not rerun; the deep baselines we did run are simple, small in number and trained on eight small datasets. *Scale.* The $O(n^3)$ eigendecomposition limits the approach to roughly 10^4^ training series on a workstation; larger problems would need a primal or iterative formulation, for which the LOO identity is still available but the cached structure differs. *Class-conditional model.* LOO-EM assumes noise that depends only on the class; it has no mechanism for instance-dependent noise. *Identifiability.* Without outside knowledge, heavy asymmetric noise cannot be told from overlapping classes, and no choice of penalty in the BIC removes that.

# 9 Conclusion

For ridge classifiers on random convolutional kernel features, the first response to suspected label noise should be regularisation, not cleaning. Symmetric noise leaves the expected decision function order-preserving and inflates its variance, the default penalty grid is too narrow to counter that, and widening it recovers up to <<DMAX('ridge_wide','ridge_std')>> accuracy points over <<N_DS>> datasets at no cost on clean labels, although under symmetric noise the gain does not extend to class-balanced metrics. Generic cleaning pipelines detect mislabelled series accurately but reduce accuracy on clean and lightly corrupted data when their detections are acted upon. An EM procedure built on closed-form leave-one-out predictions with BIC model selection is the only correction we tested that improves on a well-regularised ridge classifier under moderate to heavy class-conditional noise while staying within a few tenths of a point on clean data (calibrated confident learning matches it under heavy noise but loses about two points on clean labels), and its gains hold on accuracy, balanced accuracy and macro-F1 alike; and its limits are set by an identifiability problem rather than by computation. Future work should test these conclusions on data with verified real-world label errors and extend the leave-one-out machinery to instance-dependent noise.

# Declarations

**Funding** [To be completed by the authors.]

**Competing interests** [To be completed by the authors; e.g. "The authors have no competing interests to declare that are relevant to the content of this article."]

**Ethics approval and consent** Not applicable. The study uses only public benchmark datasets and involves no human participants or animals.

**Data availability** All datasets are publicly available from the UCR/UEA time series classification archive (https://www.timeseriesclassification.com) and are downloaded automatically by the accompanying code.

**Code availability** The complete source code, configuration, raw per-run results, and the scripts that generate every table and figure in this article are available at [repository URL to be inserted by the authors].

**Author contributions** [To be completed by the authors.]

**Use of AI tools** [To be completed by the authors in line with the publisher's policy on the disclosure of AI assistance in the preparation of the code, experiments and manuscript.]

# References

REFERENCES
