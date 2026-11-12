# Review round 7 — writing, structure, mathematics, figures, tables, references, guideline compliance

| # | Severity | Issue | Action |
|---|---|---|---|
| 1 | MAJOR | Proposition 2 originally said the arg-max is preserved "iff" T is symmetric. Only the "if" direction was proved. | Restated as "if … ; for other T it need not be", with an explicit numerical counter-example (C = 3, ρ = 0.3, g = (0.45, 0.40, 0.15) → T^T g = (0.36, 0.415, 0.225)), which was re-computed by hand and by script. |
| 2 | MAJOR | Proposition 1: constants. E[Ỹ] = aY + b11ᵀ with a = 1 − ρC/(C−1), b = ρ(2−C)/(C−1) on the ±1 scale. | Re-derived: for Y = +1, 1−2ρ = a + b ✓.; for Y = −1, −1 + 2ρ/(C−1) = −a + b ✓. A unit test (`test_symmetric_noise_expected_solution_is_scaled`) checks the affine relation numerically (written with the equivalent 0/1-scale constant). |
| 3 | MODERATE | "variance at most 1" for ε on the ±1 scale: a ±1-valued variable has variance ≤ 1 ✓. "Independent" holds across samples i, not across classes c within a sample. | Text says "independent zero-mean ε_ic" — changed to make clear independence is across samples. |
| 4 | MODERATE | Equation numbers were dropped by the DOCX writer. | Builder now appends "(n)" inside each display equation. |
| 5 | MODERATE | Minus signs: a global hyphen→minus replacement corrupted "Holm-adjusted". | Replacement restricted to numeric contexts. |
| 6 | MODERATE | Several sentences asserted orderings that depend on the data ("only LOO removal reaches parity…"). | Replaced by statements with injected numbers, or removed. |
| 7 | MODERATE | Reference style. Springer Nature journals in this field accept author–year ("Basic") style; the live guideline page could not be fetched, so numbered style may be required. | Author–year used consistently; flagged in the final risk report as a formatting item to confirm on the submission page. |
| 8 | MINOR | Inflated words. | Searched for "novel", "state-of-the-art", "significantly" (non-statistical), "outperform", "unprecedented", "revolutionary": none remain except "significant" in its statistical sense. |
| 9 | MINOR | Figures: colour is the only cue in some plots. | Markers differ per method; scatter uses a perceptually uniform colour map with a colour bar. |
| 10 | MINOR | Abstract length. | Springer guideline asks for 150–250 words; abstract trimmed to fit (see final audit for the count). |
| 11 | MINOR | Declarations. | Funding, competing interests, ethics, data, code, author contributions and AI-use statements present as placeholders/statements. |
