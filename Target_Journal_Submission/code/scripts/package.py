"""Assemble Target_Journal_Submission/ from the repository (journal_selection/ is authored in place)."""
import os, shutil, glob
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); O = os.path.join(R, "Target_Journal_Submission")
def fresh(d):
    p = os.path.join(O, d); shutil.rmtree(p, ignore_errors=True); os.makedirs(p); return p
ig = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache")
m = fresh("manuscript")
for f in ("Final_Manuscript.docx", "Final_Manuscript.pdf", "Final_Manuscript.md", "cover_letter.docx"):
    shutil.copy(os.path.join(R, "manuscript", f), m)
c = fresh("code")
for d in ("src", "experiments", "tests", "scripts", "config"):
    shutil.copytree(os.path.join(R, d), os.path.join(c, d), ignore=ig)
os.makedirs(os.path.join(c, "manuscript"))
for f in glob.glob(os.path.join(R, "manuscript", "*.py")) + [os.path.join(R, "manuscript", "manuscript_template.md"), os.path.join(R, "manuscript", "references_verified.json")]:
    shutil.copy(f, os.path.join(c, "manuscript"))
os.makedirs(os.path.join(c, "data")); shutil.copy(os.path.join(R, "data", "dataset_summary.csv"), os.path.join(c, "data"))
for f in ("run_all.py", "requirements.txt", "environment.yml", "README.md", "LICENSE"):
    shutil.copy(os.path.join(R, f), c)
r = fresh("results")
for f in ("raw_results.csv", "metrics.json", "statistics.json"):
    shutil.copy(os.path.join(R, "results", f), r)
shutil.copytree(os.path.join(R, "results", "logs"), os.path.join(r, "experiment_logs"))
open(os.path.join(r, "README.txt"), "w").write("raw_results.csv: main experiment (108 datasets, all methods).\nPer-dataset raw files of every experiment (main, BIC pass, calibrated-CL pass, large, Hydra, UEA, families, deep, sensitivity)\nare in results/raw/ of the repository.\n")
for d in ("figures", "tables", "REVIEW"):
    shutil.rmtree(os.path.join(O, d), ignore_errors=True); shutil.copytree(os.path.join(R, d), os.path.join(O, d))
s = fresh("supplementary")
for f in glob.glob(os.path.join(R, "supplementary", "*.*")):
    shutil.copy(f, s)
n = sum(len(fs) for _, _, fs in os.walk(O)); size = sum(os.path.getsize(os.path.join(a, f)) for a, _, fs in os.walk(O) for f in fs)
print(f"packaged {n} files, {size / 1e6:.0f} MB ->", O)
