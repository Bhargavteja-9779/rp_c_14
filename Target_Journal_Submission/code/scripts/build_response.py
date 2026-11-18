"""Render REVIEW/response_template.md with numbers from the result files -> REVIEW/response_to_external_review.md."""
import os, re, sys, warnings
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); warnings.filterwarnings("ignore")
from manuscript.build_manuscript import Ctx
ctx = Ctx(); ns = ctx.namespace()
src = open(os.path.join(ROOT, "REVIEW", "response_template.md"), encoding="utf-8").read()
bad = []
def ev(m):
    try:
        return str(eval(m.group(1), {"__builtins__": {}}, ns))
    except Exception as e:
        bad.append((m.group(1), repr(e))); return "??"
out = re.sub(r"<<(.+?)>>", ev, src)
open(os.path.join(ROOT, "REVIEW", "response_to_external_review.md"), "w", encoding="utf-8").write(out)
print("unresolved:", bad or "none")
