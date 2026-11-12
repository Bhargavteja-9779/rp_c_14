#!/usr/bin/env python3
"""Build the manuscript (Markdown -> DOCX -> PDF) from the template and the machine-generated results.

Every number in the text is produced by evaluating an expression against results/*.json|csv and tables/*.csv;
tables are rendered from the same files.  The script also runs the automated claim/consistency checks:
citation keys <-> reference list, figure/table numbering, and unresolved placeholders.

usage: python manuscript/build_manuscript.py [--no-pdf]
"""
import argparse, json, os, re, subprocess, sys, warnings

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from src.evaluation import aggregate as ag
from src.training import runner
from manuscript.references import REFS, cite_keys
from manuscript import sections

M = os.path.join(ROOT, "manuscript")
T = os.path.join(ROOT, "tables")


class Pt(float):
    """A number of accuracy points that prints with one decimal and survives unary minus."""
    def __neg__(self): return Pt(-float(self))
    def __str__(self): return f"{float(self):.1f}"
    __repr__ = __str__


class Ctx:
    def __init__(self):
        self.df = pd.read_csv(os.path.join(ROOT, "results", "raw_results.csv"))
        self.df["cond"] = [ag.cond_label(k, r) for k, r in zip(self.df.noise, self.df.rate)]
        self.metrics = json.load(open(os.path.join(ROOT, "results", "metrics.json")))
        self.stats = json.load(open(os.path.join(ROOT, "results", "statistics.json")))
        self.all_ds = sorted(self.df.dataset.unique())
        self.subs, self.dup = ag.subsets(self.all_ds, runner.dev_datasets(runner.load_config()))
        self._mt = {}
        self.pdt = ag.per_dataset(self.df)

    # ---- accuracy ---------------------------------------------------------------------------------
    def mt(self, subset="all", metric="acc"):
        key = (subset, metric)
        if key not in self._mt:
            self._mt[key] = ag.mean_table(self.df, sorted(self.df.method.unique()), self.subs[subset], metric)
        return self._mt[key]

    def A(self, method, cond, subset="all"):
        """Mean accuracy (%) over the datasets on which all methods are defined for the condition."""
        return f"{100 * self.mt(subset).loc[method, cond]:.1f}"

    def PTS(self, a, b, cond, subset="all"):
        return Pt(100 * (self.mt(subset).loc[a, cond] - self.mt(subset).loc[b, cond]))

    def _st(self, cand, ref, cond, subset="all"):
        rows = self.stats[f"{cand}__vs__{ref}__{subset}"]
        return next(r for r in rows if r["cond"] == cond)

    def DPTS(self, cand, ref, cond, subset="all"):
        return Pt(100 * self._st(cand, ref, cond, subset)["mean_diff"])

    def DMAX(self, cand, ref, subset="all"):
        return Pt(100 * max(r["mean_diff"] for r in self.stats[f"{cand}__vs__{ref}__{subset}"]))

    @staticmethod
    def p(v):
        return "< 0.001" if v < 0.001 else f"= {v:.3f}"

    def DSTAT(self, cand, ref, cond, subset="all"):
        r = self._st(cand, ref, cond, subset)
        return (f"{100 * r['mean_diff']:+.1f} points (95% CI {100 * r['ci_lo']:+.1f} to {100 * r['ci_hi']:+.1f}; "
                f"Holm-adjusted p {self.p(r['p_holm'])}; {r['wins']} wins, {r['losses']} losses)").replace("-0", "−0").replace("-1", "−1").replace("-2", "−2").replace("-3", "−3").replace("-4", "−4").replace("-5", "−5").replace("-6", "−6").replace("-7", "−7").replace("-8", "−8").replace("-9", "−9")

    def WL(self, cand, ref, cond, subset="all"):
        r = self._st(cand, ref, cond, subset)
        return f"{r['wins']} wins, {r['losses']} losses"

    def MD(self, metric, cand, ref, cond):
        """Mean paired difference (points) for balanced accuracy / macro-F1, all datasets."""
        r = next(r for r in self.stats[f"{metric}__{cand}__vs__{ref}__all"] if r["cond"] == cond)
        return f"{100 * r['mean_diff']:+.1f}".replace("-", "−")

    def MDP(self, metric, cand, ref, cond):
        r = next(r for r in self.stats[f"{metric}__{cand}__vs__{ref}__all"] if r["cond"] == cond)
        return f"{100 * r['mean_diff']:+.1f} points (Holm-adjusted p {self.p(r['p_holm'])}; {r['wins']} wins, {r['losses']} losses)".replace("-0", "−0").replace("-1", "−1").replace("-2", "−2").replace("-3", "−3").replace("-4", "−4")

    def IMB(self, metric, b, cond, m):
        r = next(r for r in self.metrics["by_imbalance"] if r["metric"] == metric and r["imbalance_bin"].startswith(b) and r["cond"] == cond)
        return f"{100 * r[m]:.1f}"

    def NIMB(self, b):
        return next(r for r in self.metrics["by_imbalance"] if r["imbalance_bin"].startswith(b))["n_datasets"]

    # ---- penalty ----------------------------------------------------------------------------------
    def PCT_SAT(self, cond):
        d = self.df[(self.df.method == "ridge_std") & (self.df.cond == cond)]
        return f"{100 * np.mean(np.isclose(d.alpha, 1000.0)):.0f}"

    def PCT_ABOVE(self, cond):
        d = self.df[(self.df.method == "ridge_wide") & (self.df.cond == cond)]
        return f"{100 * np.mean(d.alpha > 1000.0 * 1.001):.0f}"

    # ---- detection / EM diagnostics -----------------------------------------------------------------
    def DET(self, method, cond):
        return f"{self.metrics['detection']['det_auroc'][cond][method]:.2f}"

    def LACC(self, method, cond):
        return f"{100 * self.metrics['detection']['label_acc'][cond][method]:.1f}"

    def _diag(self, cond, method="loo_em_bic"):
        return next(r for r in self.metrics["em_diagnostics"] if r["cond"] == cond and r["method"] == method)

    def ACC_RATE(self, cond): return f"{100 * self._diag(cond)['accept_rate']:.0f}"
    def EST(self, cond): return f"{self._diag(cond)['est_noise']:.2f}"

    # ---- strata -----------------------------------------------------------------------------------
    def _bin(self, key, col, b, cond):
        return next(r for r in self.metrics[key] if r[col] == b and r["cond"] == cond)

    def BIN(self, b, cond, m): return f"{100 * self._bin('by_train_size', 'size_bin', b, cond)[m]:.1f}"
    def NBIN(self, b): return self._bin("by_train_size", "size_bin", b, "sym 0%")["n_datasets"]
    def BINDIFF(self, b, cond):
        r = self._bin("by_train_size", "size_bin", b, cond); return f"{100 * (r['loo_em_bic'] - r['ridge_wide']):+.1f}".replace("-", "−")
    def CB(self, b, cond, m): return f"{100 * self._bin('by_n_classes', 'class_bin', b, cond)[m]:.1f}"
    def NCB(self, b): return self._bin("by_n_classes", "class_bin", b, "sym 0%")["n_datasets"]
    def CBDIFF(self, b, cond):
        r = self._bin("by_n_classes", "class_bin", b, cond); return f"{100 * (r['loo_em_bic'] - r['ridge_wide']):+.1f}".replace("-", "−")

    def CORR(self, cond, k): return f"{self.metrics['extra']['error_analysis']['correlates'][cond][k]:.2f}".replace("-", "−")
    def FRAC_CHANGED(self, cond): return f"{100 * self.metrics['extra']['error_analysis']['correlates'][cond]['frac_datasets_changed']:.0f}"

    def namespace(self):
        ns = {k: getattr(self, k) for k in dir(self) if k.isupper()}
        summ = pd.read_csv(os.path.join(ROOT, "data", "dataset_summary.csv")).set_index("dataset")
        leak = pd.read_csv(os.path.join(T, "table_leakage_audit.csv"))
        ex = self.metrics.get("extra", {})
        ns.update(N_DS=len(self.all_ds), N_CONFIRM=len(self.subs["confirm"]), N_ROWS=f"{len(self.df):,}",
                  N_METHODS=self.df.method.nunique(), MED_N=int(summ.loc[self.all_ds].n_train.median()),
                  N_UEA=ex.get("uea", {}).get("n_datasets", "NA"),
                  N_DUPDS=int((leak.exact_dup_train_test > 0).sum()), N_DUP1=len(self.dup), DUPLIST=", ".join(self.dup))
        return ns


# ------------------------------------------------------------------------------------------------ tables
SHORT = "S, A and B denote symmetric, asymmetric and boundary noise at the given rate (%)"


def pct(df, dec=1):
    out = (100 * df).round(dec).map(lambda v: "–" if pd.isna(v) else f"{v:.{dec}f}")
    out.columns = [c.replace("sym ", "S ").replace("asym ", "A ").replace("boundary ", "B ").replace("aS ", "A ").replace("%", "") for c in out.columns]
    return out


def md_table(df, caption, index_name=""):
    df = df.copy()
    df.index.name = index_name
    return df.reset_index().to_markdown(index=False, disable_numparse=True) + f"\n\n: {caption}\n"


def stats_md(ctx, cand, ref, subset):
    rows = ctx.stats[f"{cand}__vs__{ref}__{subset}"]
    out = pd.DataFrame([{"Noise": r["cond"], "n": r["n"], "Mean diff.": f"{100 * r['mean_diff']:+.2f}",
                         "95% CI": f"{100 * r['ci_lo']:+.2f} to {100 * r['ci_hi']:+.2f}",
                         "p (Wilcoxon)": "<0.0001" if r['p'] < 1e-4 else f"{r['p']:.4f}", "p (Holm)": "<0.0001" if r['p_holm'] < 1e-4 else f"{r['p_holm']:.4f}",
                         "Effect (r)": f"{r['effect_rank_biserial']:+.2f}", "W/T/L": f"{r['wins']}/{r['ties']}/{r['losses']}"} for r in rows])
    return out


def build_tables(ctx):
    t = {}
    main = ctx.mt().loc[[m for m in ag.KEY]].rename(index=ag.LABEL)
    t["main"] = md_table(pct(main), f"Mean test accuracy (%) over {len(ctx.all_ds)} UCR datasets (five resamples each) on MiniRocket features. "
                         "For each column only datasets on which the condition is defined for all methods are averaged. " + SHORT, "Method")
    t["stats_wide"] = stats_md(ctx, "ridge_wide", "ridge_std", "all").to_markdown(index=False, disable_numparse=True) + \
        f"\n\n: Wide versus default penalty grid, all {len(ctx.all_ds)} datasets: paired differences in accuracy points, two-sided Wilcoxon signed-rank test, Holm correction over the twelve conditions, matched-pairs rank-biserial effect size, and wins/ties/losses over datasets\n"
    t["stats_em"] = stats_md(ctx, "loo_em_bic", "ridge_wide", "confirm").to_markdown(index=False, disable_numparse=True) + \
        f"\n\n: LOO-EM (BIC) versus wide-grid ridge on the {len(ctx.subs['confirm'])} confirmation datasets (columns as in Table 2)\n"
    abl = ctx.mt().loc[[m for m in ag.ABL if m in ctx.mt().index]].rename(index=ag.LABEL)
    t["ablation"] = md_table(pct(abl), "Ablation and variants: mean test accuracy (%) over all datasets. " + SHORT, "Variant")
    conds = ["sym 0%", "sym 30%", "asym 40%", "boundary 20%"]
    rows = []
    for key, col in (("by_train_size", "size_bin"), ("by_n_classes", "class_bin")):
        for r in ctx.metrics[key]:
            if r["cond"] in conds:
                rows.append({"Stratum": r[col].replace("<=", "≤").replace(">=", "≥"), "Noise": r["cond"], "Datasets": r["n_datasets"],
                             "Ridge (default)": f"{100 * r['ridge_std']:.1f}", "Ridge (wide)": f"{100 * r['ridge_wide']:.1f}",
                             "LOO-EM (BIC)": f"{100 * r['loo_em_bic']:.1f}", "LOO-EM v1": f"{100 * r['loo_em']:.1f}",
                             "Confident learning": f"{100 * r['conf_learning']:.1f}"})
    s = pd.DataFrame(rows)
    t["strata"] = s.to_markdown(index=False, disable_numparse=True) + "\n\n: Mean test accuracy (%) by training-set size and by number of classes\n"
    e = pd.read_csv(os.path.join(T, "table_error_analysis.csv"))
    e = e[e.cond.isin(["sym 0%", "asym 40%"])]
    e = pd.DataFrame({"Noise": e.cond, "Group": e.group, "Dataset": e.dataset, "n": e.n_train, "Classes": e.n_classes,
                      "Ridge (wide)": (100 * e.ridge_wide).map("{:.1f}".format), "LOO-EM (BIC)": (100 * e.loo_em_bic).map("{:.1f}".format),
                      "Diff.": (100 * e["diff"]).map("{:+.1f}".format)})
    t["errors"] = e.to_markdown(index=False, disable_numparse=True) + "\n\n: Datasets with the largest losses and gains of LOO-EM (BIC) relative to wide-grid ridge (accuracy %, mean of five resamples)\n"
    return t


# ------------------------------------------------------------------------------------------------ build
def render(ctx):
    src = open(os.path.join(M, "manuscript_template.md"), encoding="utf-8").read()
    tables = build_tables(ctx)
    sec = sections.build(ctx, md_table, pct)           # generated sub-sections + their tables
    for k, v in sec.items():
        src = src.replace(f"SECTION:{k}", v)
    tables.update(sections.TABLES)
    ns = ctx.namespace()
    missing = []

    def ev(m):
        try:
            return str(eval(m.group(1), {"__builtins__": {}}, ns))
        except Exception as e:  # reported, never silently left in the text
            missing.append((m.group(1), repr(e)))
            return "??"
    src = re.sub(r"<<(.+?)>>", ev, src)
    src = re.sub(r"\\tag\{(\d+)\}", r"\\qquad\\text{(\1)}", src)          # pandoc's docx writer drops \tag
    src = re.sub(r"\\mathrm\{(diag|softmax|loo)\}", r"\\text{\1}", src)
    # tables, numbered in order of appearance
    n = 0
    tabno = {}
    def tab(m):
        nonlocal n
        n += 1
        tabno[m.group(1)] = n
        body = tables[m.group(1)]
        return re.sub(r"\n: ", f"\n: **Table {n}** ", body, count=1)
    src = re.sub(r"^TABLE:(\w+)$", tab, src, flags=re.M)
    n_tab = n
    src = re.sub(r"TABREF:(\w+)", lambda m: f"Table {tabno[m.group(1)]}", src)
    # figures
    k = 0
    def fig(m):
        nonlocal k
        k += 1
        return f"![**Fig. {k}** {m.group(1)}]({os.path.join(ROOT, 'figures', m.group(2) + '.png')}){{width=6.3in}}"
    src = re.sub(r"!\[(.+?)\]\(FIG:(\w+)\)", fig, src)
    # references
    used = set(cite_keys(src))
    refs_md = "\n\n".join(REFS[key]["text"] for key in sorted(REFS, key=lambda s: REFS[s]["sort"]) if key in used)
    src = src.replace("REFERENCES", refs_md)
    problems = [f"unresolved expression {a}: {b}" for a, b in missing]
    problems += [f"citation without reference entry: {c}" for c in sorted(used - set(REFS))]
    problems += [f"reference never cited: {c}" for c in sorted(set(REFS) - used)]
    body = src.split("# References")[0]
    for i in range(1, n_tab + 1):
        if not re.search(rf"Tables? (\d+[–, and]+)*{i}\b", body.replace("**Table", "")):
            problems.append(f"Table {i} is not cited in the text")
    for i in range(1, k + 1):
        if not re.search(rf"Figs?\. (\d+[–, and]+)*{i}\b", body.replace("**Fig.", "")):
            problems.append(f"Fig. {i} is not cited in the text")
    if "SECTION:" in src or "??" in src:
        problems.append("unfilled SECTION or ?? placeholder remains")
    return src, problems, dict(tables=n_tab, figures=k, references=len(used), words=len(re.findall(r"\w+", body)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--no-pdf", action="store_true"); a = ap.parse_args()
    ctx = Ctx()
    md, problems, info = render(ctx)
    out_md = os.path.join(M, "Final_Manuscript.md")
    open(out_md, "w", encoding="utf-8").write(md)
    print(info)
    print("\n".join("PROBLEM: " + p for p in problems) or "consistency checks passed")
    from manuscript import docx_format
    docx = os.path.join(M, "Final_Manuscript.docx")
    subprocess.check_call(["pandoc", out_md, "-o", docx, "--from", "markdown+pipe_tables+table_captions+tex_math_dollars+superscript+subscript",
                           "--resource-path", ROOT])
    docx_format.polish(docx)
    if not a.no_pdf:
        subprocess.call(["soffice", "--headless", "--convert-to", "pdf", "--outdir", M, docx], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("written:", docx)
