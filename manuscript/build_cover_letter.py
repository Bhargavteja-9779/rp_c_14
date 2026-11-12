"""Cover letter (DOCX); numbers injected from the results like the manuscript."""
import os, sys, warnings
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); warnings.filterwarnings("ignore")
from docx import Document
from docx.shared import Pt, Cm
from manuscript.build_manuscript import Ctx
c = Ctx(); n = len(c.all_ds); nc = len(c.subs["confirm"])
P = [
 "[Date]",
 "The Editor-in-Chief\nJournal of King Saud University – Computer and Information Sciences\nSpringer Nature",
 "Dear Editor,",
 "We submit the manuscript “Regularise before you relabel: label noise in random convolutional kernel time series classification” for consideration as a research article.",
 f"Random-convolutional-kernel classifiers such as ROCKET and MiniRocket are widely used because they are accurate and train in seconds, yet their behaviour under mislabelled training data had not been examined systematically. Our study does so on {n} UCR datasets with five resamples each and twelve noise conditions, with additional experiments on multivariate data, a second kernel transform, larger datasets and other classifier families.",
 f"The manuscript makes three points. (1) Part of the fragility of these classifiers under label noise is caused by the customary ridge-penalty grid; widening it raises mean accuracy by up to {c.DMAX('ridge_wide','ridge_std')} points at no cost on clean labels, and we give a short proof of why symmetric noise is a variance problem for ridge classification. We also report that under symmetric noise this gain does not extend to balanced accuracy or macro-F1. (2) Generic cleaning methods detect mislabelled series well but reduce accuracy on clean and lightly corrupted data. (3) We propose LOO-EM, an expectation–maximisation procedure built on closed-form leave-one-out predictions with BIC selection of the noise model. On {nc} confirmation datasets that played no part in its design it improves on a well-regularised ridge classifier by {c.DPTS('loo_em_bic','ridge_wide','sym 30%','confirm')} points at 30% symmetric and {c.DPTS('loo_em_bic','ridge_wide','asym 40%','confirm')} points at 40% asymmetric noise, at a cost of {-c.DPTS('loo_em_bic','ridge_wide','sym 0%','confirm')} points on clean labels. These are modest effects and we present them as such.",
 "We believe the work fits the journal's scope in machine learning and data analytics and will be useful to readers who apply time series classifiers to imperfectly labelled sensor data. We have tried to be candid about limitations: all label noise is synthetic, the method does not help under instance-dependent noise, an earlier version of the method failed on many-class data and is reported in full, and published deep noisy-label methods were not rerun.",
 "The complete code, configuration, raw per-run results and the scripts that regenerate every table, figure and number in the manuscript are provided with the submission and will be made public on acceptance.",
 "The manuscript is original, has not been published, and is not under consideration elsewhere. All authors have approved the submission. The study uses only public benchmark data and involved no human participants or animals. The authors declare [no competing interests / the following competing interests: …]. [Funding statement.] [Statement on the use of AI tools in preparing the code, experiments and text, as required by the publisher's policy.]",
 "Yours sincerely,",
 "[Corresponding author name]\n[Position, Department, Institution]\n[Postal address]\n[E-mail] · [ORCID]\nOn behalf of all authors",
]
d = Document()
for s in d.sections:
    s.left_margin = s.right_margin = Cm(2.5); s.top_margin = s.bottom_margin = Cm(2.5)
for t in P:
    p = d.add_paragraph(t); p.paragraph_format.space_after = Pt(10); p.paragraph_format.line_spacing = 1.15
    for r in p.runs: r.font.name = "Times New Roman"; r.font.size = Pt(11.5)
out = os.path.join(ROOT, "manuscript", "cover_letter.docx"); d.save(out); print("written", out)
