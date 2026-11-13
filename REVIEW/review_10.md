# Review round 10 — final adversarial review (aim: find a reason to reject)

Read against the final build (30 pages, 10 tables, 10 figures, 42 references, 144,785 result rows).

## Is there a fatal flaw?

One candidate was examined in depth: *"the proposed method was designed after seeing held-out results, so its evaluation is circular."* It is not fatal because the primary LOO-EM comparison uses 60 datasets that were in neither the development nor the design set, the freeze preceded their inspection (commit history and `supplementary/protocol.md`), and the "all datasets" figures are labelled as optimistic. No fatal flaw was found. The paper's main risk is not error but modest novelty.

## Major weaknesses (real ones only — six, not ten)

1. **Small effect of the proposed method.** +1.3 and +2.2 points at two of twelve conditions on the confirmation set; a significant −0.3 on clean labels; statistically indistinguishable from calibrated confident learning at ≥ 20 % noise. *Fix:* none possible without overclaiming; text states all of this.
2. **Synthetic noise only.** *Fix:* not possible with public data known to us; disclosed.
3. **No rerun of published deep noisy-label TSC methods; on the six small datasets Co-teaching with the true noise rate beats every ridge variant at 30 % symmetric noise (82.0 % vs 78.1 %).** *Fix:* the result is reported in Sect. 6.6 and no claim against deep methods is made.
4. **The headline recommendation is metric-dependent.** *Fix:* stated in abstract, Sect. 6.1, Discussion, Conclusion.
5. **Secondary experiments are thinner than planned:** Hydra on 29 datasets (one excluded for memory), no MultiRocket, three large datasets instead of four, seeds 0–2. *Fix:* each limitation is stated where the result is reported; nothing is claimed for experiments that were not run.
6. **Journal guideline compliance could not be verified against the live page.** *Fix:* listed as a pre-submission task.

## Moderate weaknesses (nine)

1. Hydra: the wide grid costs 0.8 points on clean labels (not significant, 6 wins / 4 losses) — "no cost on clean labels" is established for MiniRocket only. Text for Hydra reports the number.
2. LOO-EM hits the 20-iteration cap in most runs; shown not to matter (5 vs 50 iterations) but convergence is not demonstrated.
3. The noise-rate estimate is biased low (0.15 at 0.20); the paper says "conservative" and does not recommend it as an estimator.
4. Holm correction is per method pair; with about twenty pairs reported, some supplementary "significant" results would not survive a global correction. Conclusions rest on pre-specified pairs.
5. Per-condition n differs (conditions undefined when a class vanishes), so columns of Table 1 are not strictly comparable across noise rates for tiny datasets.
6. Timings come from a shared VM with background load.
7. Boundary noise is generated from the same model family that is then evaluated; it is adversarial to margin methods by construction, which is stated, but it is one particular instance-dependent model.
8. The deep baselines have no validation-based early stopping (none is possible without clean labels) and fixed hyper-parameters.
9. Reference list: seven entries carry caveats (see `citation_audit.md`).

## Minor

* LibreOffice renders `diag`, `softmax` with letter-spacing in the PDF; Word renders the DOCX equations correctly. Submit the DOCX.
* Table 1 and Table 4 are dense (12 columns); acceptable in a single-column manuscript, may need landscape in typesetting.
* British spelling is used consistently; the journal accepts either.
* Author, affiliation, funding, contribution and AI-use fields are placeholders.

## Fixes applied in this round

* Statements about MultiRocket, ElectricDevices, "eight" deep datasets and a 8,926-series scalability run were removed or corrected to what was actually run.
* Sensitivity text corrected: a strong transition-matrix prior *does* lower accuracy under asymmetric noise (62.9 % vs 64.1 %); the earlier sentence said the prior strength was immaterial.
* Co-teaching's win at 30 % symmetric noise is now stated explicitly rather than left to the table.

Recommendation: acceptable for submission as an honest empirical study; outcome of external review uncertain.
