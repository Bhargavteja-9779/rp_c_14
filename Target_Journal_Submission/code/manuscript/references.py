"""Reference list (Springer Basic author-year style).  Every entry was checked against Crossref (DOI), arXiv, or the
publisher/proceedings page; see REVIEW/citation_audit.md.  `pat` is the in-text form used for the automatic
citation <-> reference cross-check."""
import re

def R(pat, text, sort=None):
    return dict(pat=pat, text=text, sort=sort or text.lower())

REFS = {
 "angluin1988": R("Angluin and Laird 1988", "Angluin D, Laird P (1988) Learning from noisy examples. Mach Learn 2:343–370. https://doi.org/10.1023/A:1022873112823"),
 "atkinson2020": R("Atkinson and Metsis 2020", "Atkinson G, Metsis V (2020) Identifying label noise in time-series datasets. In: Adjunct proceedings of the 2020 ACM international joint conference on pervasive and ubiquitous computing (UbiComp-ISWC '20). ACM, pp 238–243. https://doi.org/10.1145/3410530.3414366"),
 "bagnall2017": R("Bagnall et al. 2017", "Bagnall A, Lines J, Bostrom A, Large J, Keogh E (2017) The great time series classification bake off: a review and experimental evaluation of recent algorithmic advances. Data Min Knowl Discov 31:606–660. https://doi.org/10.1007/s10618-016-0483-9"),
 "bartlett2020": R("Bartlett et al. 2020", "Bartlett PL, Long PM, Lugosi G, Tsigler A (2020) Benign overfitting in linear regression. Proc Natl Acad Sci USA 117:30063–30070. https://doi.org/10.1073/pnas.1907378117"),
 "bootkrajang2012": R("Bootkrajang and Kabán 2012", "Bootkrajang J, Kabán A (2012) Label-noise robust logistic regression and its applications. In: Machine learning and knowledge discovery in databases (ECML PKDD 2012). Lecture notes in computer science, vol 7523. Springer, pp 143–158. https://doi.org/10.1007/978-3-642-33460-3_15"),
 "brodley1999": R("Brodley and Friedl 1999", "Brodley CE, Friedl MA (1999) Identifying mislabeled training data. J Artif Intell Res 11:131–167. https://doi.org/10.1613/jair.606"),
 "castellani2021": R("Castellani et al. 2021", "Castellani A, Schmitt S, Hammer B (2021) Estimating the electrical power output of industrial devices with end-to-end time-series classification in the presence of label noise. In: Machine learning and knowledge discovery in databases. Research track (ECML PKDD 2021). Lecture notes in computer science, vol 12975. Springer, pp 469–484. https://doi.org/10.1007/978-3-030-86486-6_29"),
 "chen2021": R("Chen et al. 2021", "Chen P, Ye J, Chen G, Zhao J, Heng PA (2021) Robustness of accuracy metric and its inspirations in learning with noisy labels. Proc AAAI Conf Artif Intell 35(13):11451–11461. https://doi.org/10.1609/aaai.v35i13.17364"),
 "dau2019": R("Dau et al. 2019", "Dau HA, Bagnall A, Kamgar K, Yeh CCM, Zhu Y, Gharghabi S, Ratanamahatana CA, Keogh E (2019) The UCR time series archive. IEEE/CAA J Autom Sin 6:1293–1305. https://doi.org/10.1109/JAS.2019.1911747"),
 "dawid1979": R("Dawid and Skene 1979", "Dawid AP, Skene AM (1979) Maximum likelihood estimation of observer error-rates using the EM algorithm. J R Stat Soc Ser C Appl Stat 28:20–28. https://doi.org/10.2307/2346806"),
 "dempster2020": R("Dempster et al. 2020", "Dempster A, Petitjean F, Webb GI (2020) ROCKET: exceptionally fast and accurate time series classification using random convolutional kernels. Data Min Knowl Discov 34:1454–1495. https://doi.org/10.1007/s10618-020-00701-z"),
 "dempster2021": R("Dempster et al. 2021", "Dempster A, Schmidt DF, Webb GI (2021) MiniRocket: a very fast (almost) deterministic transform for time series classification. In: Proceedings of the 27th ACM SIGKDD conference on knowledge discovery and data mining. ACM, pp 248–257. https://doi.org/10.1145/3447548.3467231"),
 "dempster2023": R("Dempster et al. 2023", "Dempster A, Schmidt DF, Webb GI (2023) Hydra: competing convolutional kernels for fast and accurate time series classification. Data Min Knowl Discov 37:1779–1805. https://doi.org/10.1007/s10618-023-00939-3"),
 "dempster2024a": R("Dempster et al. 2024a", "Dempster A, Schmidt DF, Webb GI (2024a) quant: a minimalist interval method for time series classification. Data Min Knowl Discov 38:2377–2402. https://doi.org/10.1007/s10618-024-01036-9"),
 "dempster2024b": R("Dempster et al. 2024b", "Dempster A, Webb GI, Schmidt DF (2024b) Prevalidated ridge regression is a highly-efficient drop-in replacement for logistic regression for high-dimensional data. arXiv preprint arXiv:2401.15610"),
 "demsar2006": R("Demšar 2006", "Demšar J (2006) Statistical comparisons of classifiers over multiple data sets. J Mach Learn Res 7:1–30", "demsar"),
 "frenay2014": R("Frénay and Verleysen 2014", "Frénay B, Verleysen M (2014) Classification in the presence of label noise: a survey. IEEE Trans Neural Netw Learn Syst 25:845–869. https://doi.org/10.1109/TNNLS.2013.2292894", "frenay"),
 "ghosh2017": R("Ghosh et al. 2017", "Ghosh A, Kumar H, Sastry PS (2017) Robust loss functions under label noise for deep neural networks. Proc AAAI Conf Artif Intell 31(1). https://doi.org/10.1609/aaai.v31i1.10894"),
 "golub1979": R("Golub et al. 1979", "Golub GH, Heath M, Wahba G (1979) Generalized cross-validation as a method for choosing a good ridge parameter. Technometrics 21:215–223. https://doi.org/10.1080/00401706.1979.10489751"),
 "han2018": R("Han et al. 2018", "Han B, Yao Q, Yu X, Niu G, Xu M, Hu W, Tsang I, Sugiyama M (2018) Co-teaching: robust training of deep neural networks with extremely noisy labels. In: Advances in neural information processing systems 31 (NeurIPS 2018). arXiv:1804.06872"),
 "hastie2009": R("Hastie et al. 2009", "Hastie T, Tibshirani R, Friedman J (2009) The elements of statistical learning: data mining, inference, and prediction, 2nd edn. Springer, New York. https://doi.org/10.1007/978-0-387-84858-7"),
 "hoerl1970": R("Hoerl and Kennard 1970", "Hoerl AE, Kennard RW (1970) Ridge regression: biased estimation for nonorthogonal problems. Technometrics 12:55–67. https://doi.org/10.1080/00401706.1970.10488634"),
 "holm1979": R("Holm 1979", "Holm S (1979) A simple sequentially rejective multiple test procedure. Scand J Stat 6:65–70"),
 "fawaz2019": R("Ismail Fawaz et al. 2019", "Ismail Fawaz H, Forestier G, Weber J, Idoumghar L, Muller PA (2019) Deep learning for time series classification: a review. Data Min Knowl Discov 33:917–963. https://doi.org/10.1007/s10618-019-00619-1"),
 "fawaz2020": R("Ismail Fawaz et al. 2020", "Ismail Fawaz H, Lucas B, Forestier G, Pelletier C, Schmidt DF, Weber J, Webb GI, Idoumghar L, Muller PA, Petitjean F (2020) InceptionTime: finding AlexNet for time series classification. Data Min Knowl Discov 34:1936–1962. https://doi.org/10.1007/s10618-020-00710-y"),
 "li2026": R("Li et al. 2026", "Li Z, Liu Y, Tanaka G (2026) Robust and efficient noisy-label time-series classification via dynamic time warping based granular ball computing. arXiv preprint arXiv:2608.11704"),
 "liu2016": R("Liu and Tao 2016", "Liu T, Tao D (2016) Classification with noisy labels by importance reweighting. IEEE Trans Pattern Anal Mach Intell 38:447–461. https://doi.org/10.1109/TPAMI.2015.2456899"),
 "liu2023": R("Liu et al. 2023", "Liu Z, Ma P, Chen D, Pei W, Ma Q (2023) Scale-teaching: robust multi-scale training for time series classification with noisy labels. In: Advances in neural information processing systems 36 (NeurIPS 2023)"),
 "ma2023": R("Ma et al. 2023", "Ma P, Liu Z, Zheng J, Wang L, Ma Q (2023) CTW: confident time-warping for time-series label-noise learning. In: Proceedings of the thirty-second international joint conference on artificial intelligence (IJCAI-23), pp 4046–4054. https://doi.org/10.24963/ijcai.2023/450"),
 "middlehurst2024a": R("Middlehurst et al. 2024a", "Middlehurst M, Schäfer P, Bagnall A (2024a) Bake off redux: a review and experimental evaluation of recent time series classification algorithms. Data Min Knowl Discov 38:1958–2031. https://doi.org/10.1007/s10618-024-01022-1"),
 "middlehurst2024b": R("Middlehurst et al. 2024b", "Middlehurst M, Ismail-Fawaz A, Guillaume A, Holder C, Guijo-Rubio D, Bulatova G, Tsaprounis L, Mentel L, Walter M, Schäfer P, Bagnall A (2024b) aeon: a Python toolkit for learning from time series. arXiv preprint arXiv:2406.14231"),
 "natarajan2013": R("Natarajan et al. 2013", "Natarajan N, Dhillon IS, Ravikumar P, Tewari A (2013) Learning with noisy labels. In: Advances in neural information processing systems 26 (NIPS 2013)"),
 "northcutt2021": R("Northcutt et al. 2021", "Northcutt C, Jiang L, Chuang I (2021) Confident learning: estimating uncertainty in dataset labels. J Artif Intell Res 70:1373–1411. https://doi.org/10.1613/jair.1.12125"),
 "patrini2017": R("Patrini et al. 2017", "Patrini G, Rozza A, Menon AK, Nock R, Qu L (2017) Making deep neural networks robust to label noise: a loss correction approach. In: 2017 IEEE conference on computer vision and pattern recognition (CVPR), pp 2233–2241. https://doi.org/10.1109/CVPR.2017.240"),
 "schwarz1978": R("Schwarz 1978", "Schwarz G (1978) Estimating the dimension of a model. Ann Stat 6:461–464. https://doi.org/10.1214/aos/1176344136"),
 "song2023": R("Song et al. 2023", "Song H, Kim M, Park D, Shin Y, Lee JG (2023) Learning from noisy labels with deep neural networks: a survey. IEEE Trans Neural Netw Learn Syst 34:8135–8153. https://doi.org/10.1109/TNNLS.2022.3152527"),
 "tan2022": R("Tan et al. 2022", "Tan CW, Dempster A, Bergmeir C, Webb GI (2022) MultiRocket: multiple pooling operators and transformations for fast and effective time series classification. Data Min Knowl Discov 36:1623–1646. https://doi.org/10.1007/s10618-022-00844-1"),
 "tibshirani2002": R("Tibshirani and Efron 2002", "Tibshirani RJ, Efron B (2002) Pre-validation and inference in microarrays. Stat Appl Genet Mol Biol 1(1):Article 1. https://doi.org/10.2202/1544-6115.1000"),
 "wang2017": R("Wang et al. 2017", "Wang Z, Yan W, Oates T (2017) Time series classification from scratch with deep neural networks: a strong baseline. In: 2017 international joint conference on neural networks (IJCNN), pp 1578–1585. https://doi.org/10.1109/IJCNN.2017.7966039"),
 "wilcoxon1945": R("Wilcoxon 1945", "Wilcoxon F (1945) Individual comparisons by ranking methods. Biometrics Bull 1:80–83. https://doi.org/10.2307/3001968"),
 "wilson1972": R("Wilson 1972", "Wilson DL (1972) Asymptotic properties of nearest neighbor rules using edited data. IEEE Trans Syst Man Cybern SMC-2:408–421. https://doi.org/10.1109/TSMC.1972.4309137"),
 "zhang2018": R("Zhang and Sabuncu 2018", "Zhang Z, Sabuncu MR (2018) Generalized cross entropy loss for training deep neural networks with noisy labels. In: Advances in neural information processing systems 31 (NeurIPS 2018). arXiv:1805.07836"),
}

_CITE = re.compile(r"((?:[A-ZÀ-Ý][\w\-éèšáñ]+(?: [A-Z][\w\-éèšáñ]+)?)(?: et al\.| and [A-ZÀ-Ý][\w\-éèšáñ]+)?),? \(?((?:19|20)\d\d[ab]?)\)?")


def cite_keys(text):
    """Keys of all references cited in the text; unknown author-year strings are returned as '?:<string>'."""
    body = text.split("# References")[0]
    pats = {v["pat"]: k for k, v in REFS.items()}
    keys = [k for p, k in pats.items() if re.search(re.escape(p).replace(r"\ (?=\d)", " "), body) or
            re.search(re.escape(p.rsplit(" ", 1)[0]) + r",? \(" + re.escape(p.rsplit(" ", 1)[1]) + r"\)", body)]
    for m in _CITE.finditer(body):
        s = f"{m.group(1)} {m.group(2)}"
        if s not in pats and not any(s.endswith(p) for p in pats) and m.group(1).split()[0] not in (
                "Table", "Fig", "Sect", "Eq", "Seed", "Python", "Proposition", "In", "The", "On", "At", "With", "For", "Over", "Under"):
            keys.append("?:" + s)
    return keys
