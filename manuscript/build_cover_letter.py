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
 "We submit the manuscript “Regularise before you relabel: random convolutional kernel time series classification under synthetic and real annotation noise” for consideration as a research article.",
 f"Random-convolutional-kernel classifiers such as ROCKET and MiniRocket are widely used because they are accurate and train in seconds, yet their behaviour under mislabelled training data had not been examined, and noisy-label methods for time series are almost always evaluated with synthetic noise. Our study combines a short theory for ridge classifiers under label noise, a benchmark on {n} UCR datasets with five resamples and twelve noise conditions, a low-cost label-correction method (LOO-EM) built on closed-form leave-one-out predictions, and a pre-registered evaluation on real annotation errors: {c.CINCX('n_records')} ECG recordings whose original operational labels were later revised by experts.",
 f"The main finding is that conclusions drawn from synthetic noise did not transfer to real errors. With synthetic noise, a wider penalty grid and LOO-EM improved accuracy (the wider grid by up to {c.DMAX('ridge_wide','ridge_std')} points over all datasets, LOO-EM by a further {c.DPTS('loo_em_bic','ridge_wide','asym 40%','confirm')} points at 40% asymmetric noise on held-out datasets). On the real errors, training on the noisy instead of the expert labels cost only {c.CINCACC('ridge_wide_v3labels','ridge_wide')} points, no relabelling or removal method improved agreement with the expert labels, and LOO-EM, our own method, lowered it by {-c.CINCACC('loo_em_bic','ridge_wide')} points. We report this result in full, explain it, and turn it into practical guidance: choose the regularisation with the target metric in mind, and use label-error rankings to prioritise expert review rather than to relabel automatically.",
 "We believe the work fits the journal's scope in machine learning and data analytics and will be useful to readers who train time series classifiers on imperfect operational labels. The manuscript is explicit about its limitations: the real-noise evidence comes from one public dataset whose revised labels are themselves imperfect, only one time-series-specific deep method (re-implemented) was compared, and all experiments ran on CPUs.",
 "All code, configuration, raw per-run results and the scripts that regenerate every table, figure and number in the manuscript are provided with the submission and will be made public on acceptance. The analysis protocols, including those written during revision, are in the supplementary material.",
 "The manuscript is original, has not been published, and is not under consideration elsewhere. All authors have approved the submission. The study uses only public data (the UCR/UEA archive and PhysioNet); no new data were collected and no human participants or animals were involved. The authors declare [no competing interests / the following competing interests: …]. [Funding statement.] [Statement on the use of AI tools in preparing the code, experiments and text, as required by the publisher's policy.]",
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
