# Journal comparison (candidates actually investigated)

Submission-to-acceptance medians were **measured**, not copied from marketing pages: for each journal we
downloaded the "received" and "accepted" dates of recent articles (published from March 2026 onward) from
Crossref assertions or PubMed history records and computed the distribution (`scripts/journal/cr.py`, `pm.py`).
Publishers that do not deposit these dates (Elsevier, IEEE, OUP on Crossref, T&F in part) could only be measured
when the journal is PubMed-indexed; otherwise they are marked "not measurable".
Impact factors/quartiles in this table are indicative and were **not** checked in Clarivate JCR.

| Journal (publisher) | n | Median days | Share ≤ 72 d | Decision |
|---|---|---|---|---|
| **J. King Saud Univ. – Comput. Inf. Sci. (Springer Nature)** | 199 (904 for all 2026) | **69 (68)** | 0.53 (0.56) | **Selected** – only CS/AI journal under the 72-day limit |
| J. Phys. Chem. Letters (ACS) | 299 | 51 | 0.79 | Rejected – scope (physical chemistry letters) |
| Environ. Sci. Technol. Letters (ACS) | 18 | 42 | 0.78 | Rejected – scope |
| Biology Methods & Protocols (OUP) | 32 | 57 | 0.56 | Rejected – scope, small sample, Q1 not established |
| JACS Au (ACS) | 299 | 60 | 0.69 | Rejected – scope |
| Optics Express / Biomed. Opt. Express (Optica) | 278 / 258 | 62 / 64 | 0.61 | Rejected – scope |
| J. Comput. Chem. (Wiley) | 168 | 71 | 0.51 | Rejected – scope |
| J. Chem. Theory Comput. (ACS) | 296 | 79 | 0.39 | Rejected – over limit, scope |
| J. Chem. Inf. Model. (ACS) | 297 | 88 | 0.33 | Rejected – over limit |
| Int. J. Med. Inform. (Elsevier) | 300 | 88 | 0.37 | Rejected – over limit, scope |
| Comput. Struct. Biotechnol. J. (Elsevier) | 221 | 91 | 0.35 | Rejected – over limit |
| J. Comput. Design Eng. (OUP) | – | 105 (self-reported "15 weeks") | – | Rejected – over limit |
| JAMIA (OUP) | 193 | 115 | 0.20 | Rejected – over limit |
| Briefings in Bioinformatics (OUP) | 299 | 120 | 0.13 | Rejected – over limit |
| Comput. Biol. Med. (Elsevier) | 288 | 121 | 0.22 | Rejected – over limit |
| National Science Review (OUP) | 213 | 122 | 0.15 | Rejected – over limit |
| Eng. Appl. Comput. Fluid Mech. (T&F) | 163 | 123 | 0.16 | Rejected |
| Mach. Learn.: Sci. Technol. (IOP) | 185 | 125 | 0.11 | Rejected – over limit (good scope fit) |
| Scientific Reports (Springer Nature) | 190 | 126 | 0.17 | Rejected – over limit |
| BMC Bioinformatics | 196 | 126 | 0.13 | Rejected |
| Advanced Intelligent Systems (Wiley) | 172 | 128 | 0.15 | Rejected – over limit |
| GIScience & Remote Sensing (T&F) | 182 | 130 | 0.06 | Rejected |
| Engineering Reports (Wiley) | 193 | 131 | 0.16 | Rejected |
| J. Cheminformatics (Springer) | 104 | 132 | 0.08 | Rejected |
| Complex & Intelligent Systems (Springer) | 199 | 150 | 0.08 | Rejected – over limit (good scope fit) |
| Bioinformatics (OUP) | 262 | 152 | 0.07 | Rejected |
| Int. J. Comput. Intell. Syst. (Springer) | 196 | 154 | 0.11 | Rejected |
| CAAI Trans. Intell. Technol. (Wiley/IET) | 84 | 161 | 0.11 | Rejected |
| Journal of Cloud Computing (Springer) | 151 | 161 | 0.27 | Rejected |
| Expert Systems (Wiley) | 194 | 163 | 0.10 | Rejected |
| PNAS Nexus (OUP) | 298 | 168 | 0.11 | Rejected |
| npj Digital Medicine | 198 | 186 | 0.06 | Rejected |
| Artificial Intelligence Review (Springer) | 199 | 194 | 0.06 | Rejected |
| Neural Networks (Elsevier) | 300 | 222 | 0.00 | Rejected |
| Journal of Big Data (Springer) | 183 | 257 | 0.02 | Rejected |
| ICT Express, Results in Engineering, Alexandria Eng. J., Egyptian Informatics J., Ain Shams Eng. J., JESTECH, Array, Energy and AI, Internet of Things (all Elsevier); IEEE letters journals; PeerJ CS | – | **not measurable** (no deposited dates; publisher pages bot-protected) | – | Not selected – acceptance time could not be verified, as the brief requires |

About 100 journals were measured in total (full lists: `scripts/journal/l1.txt` … `l4.txt`).
