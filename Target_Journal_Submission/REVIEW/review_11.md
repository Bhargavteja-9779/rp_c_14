# Review round 11 — external review response check (does the revision answer the professor's report?)

Checked item by item against `response_to_external_review.md`.

| Issue in external review | Answered by | Residual weakness | Severity |
|---|---|---|---|
| 1 Real-world noisy labels | PhysioNet/CinC 2017, 8,528 ECGs, original vs expert-revised labels, pre-registered protocol | One dataset; reference labels imperfect; no industrial data | MAJOR (disclosed) |
| 2 Modest LOO-EM gains | Repositioned; real data show it can hurt; decision guidance in Sects. 4.5 and 7 | The "proposed method" is now a qualified contribution | MODERATE |
| 3 Deep/TS-specific baselines | CTW re-implemented (7 datasets), backward correction (108 + ECG), FCN/GCE/Co-teaching (6) | No SREA/Scale-teaching; CPU-only, untuned | MAJOR (disclosed) |
| 4 Scalability | Primal route + scaling experiment with peak memory | Dual route cubic; no streaming implementation | MINOR |
| 5 Instance-dependent noise | Candidate model with pre-set adoption rule → rejected | Remains unsolved | MODERATE (disclosed) |
| 6 Deployment story | Deployment figure with measured latency/size/memory; review-list workflow | Not a field deployment | MINOR |
| Standardisation in LOO | Exact-LOO sensitivity on 41 datasets | Transform not refitted per left-out series (shared with all LOO pipelines) | MINOR |
| Class-wise metrics, confusion matrices, calibration | Class-wise/calibration pass on all datasets; ECG confusion matrices | — | fixed |
| Title suggestion ("industrial") | Not adopted, reason given | — | — |

No point of the external review is left unanswered. Two cannot be fully satisfied with public data and a CPU:
industrial ground truth and GPU-scale deep baselines.
