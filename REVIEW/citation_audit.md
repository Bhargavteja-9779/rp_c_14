# Citation audit

Every reference in `manuscript/references.py` was checked for author list, title, year, venue, volume/pages and DOI.

| How verified | References |
|---|---|
| Crossref REST API by DOI (`scripts/verify_refs.py`, output in `manuscript/references_verified.json`) | Angluin & Laird 1988; Bagnall et al. 2017; Bartlett et al. 2020; Bootkrajang & Kabán 2012; Brodley & Friedl 1999; Castellani et al. 2021; Chen et al. 2021; Dau et al. 2019; Dawid & Skene 1979; Dempster et al. 2020, 2021, 2023, 2024a; Frénay & Verleysen 2014; Ghosh et al. 2017; Golub et al. 1979; Hastie et al. 2009; Hoerl & Kennard 1970; Ismail Fawaz et al. 2019, 2020; Liu & Tao 2016; Ma et al. 2023; Middlehurst et al. 2024a; Northcutt et al. 2021; Patrini et al. 2017; Schwarz 1978; Song et al. 2023; Tan et al. 2022; Tibshirani & Efron 2002; Wang et al. 2017; Wilcoxon 1945; Wilson 1972 |
| arXiv API | Dempster et al. 2024b (2401.15610); Li et al. 2026 (2608.11704); Middlehurst et al. 2024b (2406.14231); Han et al. 2018 (1804.06872); Zhang & Sabuncu 2018 (1805.07836) |
| Proceedings / journal page via web search | Liu et al. 2023 (NeurIPS 2023 poster page and proceedings PDF); Atkinson & Metsis 2020 (ACM DOI 10.1145/3410530.3414366); Demšar 2006 (JMLR 7:1–30); Holm 1979 (Scand J Stat 6:65–70, JSTOR 4615733); Natarajan et al. 2013 (NeurIPS proceedings page) |

**Items an author should double-check before submission**

* Dawid & Skene (1979): Crossref gives first page 20; the end page 28 is from memory of the journal issue, not from Crossref.
* Wilcoxon (1945): end page 83 likewise.
* Atkinson & Metsis (2020): one index gives pages 238–243; the ACM PDF is six pages. Kept as 238–243.
* Liu & Tao (2016): the DOI record was retrieved once (title, authors, volume 38, pages 447–461); a second request was rate-limited.
* Natarajan et al. (2013): page numbers were removed from the entry because they could not be verified.
* Middlehurst et al. (2024b): cited as the arXiv preprint; a JMLR version exists according to the authors but was not verified here.
* Li et al. (2026) and Dempster et al. (2024b) are preprints; check for published versions.

**Claims attributed to references — spot checks**

* "default grid of ten penalties between 1e-3 and 1e3" — matches the aeon implementation used (`RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))`) and the ROCKET paper's description.
* Chen et al. (2021) — title and abstract support "accuracy on noisy validation labels is informative under symmetric noise"; the affine relation is stated in our own words.
* Li et al. (2026) — only the abstract was read (DTW granular balls, symmetric noise, four datasets); the manuscript says no more than that.
* TS-ACL (arXiv 2410.15954) was read during topic selection and is not cited in the manuscript (different topic).

**Automatic cross-check.** `build_manuscript.py` fails the consistency check if an in-text author–year string has no entry or an entry is never cited; the current build reports 42 references, all cited.
