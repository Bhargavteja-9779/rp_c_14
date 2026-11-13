"""Sub-sections whose wording depends on which secondary experiments exist; all numbers come from
results/metrics.json ('extra') and results/statistics.json ('secondary')."""
import pandas as pd

from src.evaluation import aggregate as ag

TABLES = {}
CONDS = ["sym 0%", "sym 20%", "sym 40%", "asym 20%", "asym 40%", "boundary 20%"]
NAMES = {"ridge_std": "Ridge (default grid)", "ridge_wide": "Ridge (wide grid)", "loo_em_bic": "LOO-EM (BIC)", "loo_em": "LOO-EM v1",
         "conf_learning": "Confident learning (uncalibrated)", "conf_learning_cal": "Confident learning (calibrated)", "em_sym_T": "Symmetric T only", "enn": "ENN", "loo_relabel": "LOO relabel",
         "quant": "QUANT", "nn_ed": "1-NN Euclidean", "fcn_ce": "FCN, cross-entropy", "fcn_gce": "FCN, GCE loss",
         "fcn_coteach": "FCN, Co-teaching (true rate given)"}


def _rows(tag, acc, methods, conds=CONDS):
    out = []
    for m in methods:
        if all(m in acc.get(c, {}) for c in conds if c in acc):
            out.append({"Experiment": tag, "Method": NAMES.get(m, m),
                        **{c: ("–" if c not in acc or acc[c].get(m) is None or acc[c][m] != acc[c][m] else f"{100 * acc[c][m]:.1f}") for c in conds}})
    return out


def _st(ctx, exp, cand, ref, cond):
    for r in ctx.stats.get("secondary", []):
        if r["experiment"] == exp and r["candidate"] == cand and r["reference"] == ref and r["cond"] == cond:
            return r
    return None


def _d(ctx, exp, cand, ref, cond):
    r = _st(ctx, exp, cand, ref, cond)
    if r is None:
        return "n/a"
    p = "< 0.001" if r["p_holm"] < 0.001 else f"= {r['p_holm']:.3f}"
    return f"{100 * r['mean_diff']:+.1f} points (Holm-adjusted p {p}; {r['wins']} wins, {r['losses']} losses)".replace("-", "−").replace("Holm−", "Holm-")


def build(ctx, md_table, pct):
    ex = ctx.metrics.get("extra", {})
    sec = {}
    # ------------------------------------------------------------------ generalisation
    rows, parts = [], []
    for key, tag in (("transform_hydra", "Hydra"), ("transform_multirocket", "MultiRocket")):
        if key in ex:
            e = key.split("_")[1]
            rows += _rows(f"{tag} ({ex[key]['n_datasets']} UCR datasets)", ex[key]["mean_accuracy"], ["ridge_std", "ridge_wide", "loo_em_bic", "loo_em", "conf_learning"])
            note = " The datasets are the fixed random subset of the sensitivity study, without StarLightCurves, whose Hydra transform exceeded the available memory. Hydra counts are square-rooted and standardised here rather than passed through the sparse scaler of the original pipeline." if tag == "Hydra" else ""
            parts.append(f"*{tag} features* ({ex[key]['n_datasets']} datasets, seeds 0–2).{note} The wide grid changes accuracy relative to the default grid by "
                         f"{_d(ctx, e, 'ridge_wide', 'ridge_std', 'sym 0%')} on clean labels, {_d(ctx, e, 'ridge_wide', 'ridge_std', 'sym 40%')} at 40% symmetric and "
                         f"{_d(ctx, e, 'ridge_wide', 'ridge_std', 'asym 40%')} at 40% asymmetric noise. LOO-EM relative to wide-grid ridge: "
                         f"{_d(ctx, e, 'loo_em_bic', 'ridge_wide', 'sym 0%')} on clean labels, {_d(ctx, e, 'loo_em_bic', 'ridge_wide', 'sym 40%')} at 40% symmetric and "
                         f"{_d(ctx, e, 'loo_em_bic', 'ridge_wide', 'asym 40%')} at 40% asymmetric noise.")
    if "uea" in ex:
        rows += _rows(f"MiniRocket, multivariate ({ex['uea']['n_datasets']} UEA datasets)", ex["uea"]["mean_accuracy"],
                      ["ridge_std", "ridge_wide", "loo_em_bic", "loo_em", "conf_learning"])
        parts.append(f"*Multivariate series* ({ex['uea']['n_datasets']} UEA datasets, seeds 0–2). Wide versus default grid: "
                     f"{_d(ctx, 'uea', 'ridge_wide', 'ridge_std', 'sym 40%')} at 40% symmetric and {_d(ctx, 'uea', 'ridge_wide', 'ridge_std', 'asym 40%')} at 40% asymmetric noise. "
                     f"LOO-EM versus wide-grid ridge: {_d(ctx, 'uea', 'loo_em_bic', 'ridge_wide', 'sym 0%')} on clean labels and "
                     f"{_d(ctx, 'uea', 'loo_em_bic', 'ridge_wide', 'asym 40%')} at 40% asymmetric noise. With this few datasets the tests have little power and the "
                     f"table should be read descriptively.")
    if "large" in ex:
        ds = ex["large"]["datasets"]
        rows += _rows(f"MiniRocket, largest UCR datasets ({len(ds)})", ex["large"]["mean_accuracy"], ["ridge_std", "ridge_wide", "loo_em_bic", "loo_em", "enn"])
        a = ex["large"]["mean_accuracy"]
        g = lambda m, c: f"{100 * a[c][m]:.1f}"
        parts.append(f"*Larger training sets* ({', '.join(ds)}; 3,601–7,200 training series, seeds 0–2; no significance tests with {len(ds)} datasets). "
                     f"At 40% asymmetric noise the default grid gives {g('ridge_std', 'asym 40%')}%, the wide grid {g('ridge_wide', 'asym 40%')}%, LOO-EM {g('loo_em_bic', 'asym 40%')}% "
                     f"and v1 {g('loo_em', 'asym 40%')}%; at 40% symmetric noise {g('ridge_std', 'sym 40%')}%, {g('ridge_wide', 'sym 40%')}%, {g('loo_em_bic', 'sym 40%')}% and {g('loo_em', 'sym 40%')}%; "
                     f"on clean labels {g('ridge_std', 'sym 0%')}%, {g('ridge_wide', 'sym 0%')}%, {g('loo_em_bic', 'sym 0%')}% and {g('loo_em', 'sym 0%')}%.")
    text = "TABREF:general summarises three checks of external validity for the kernel-transform pipeline.\n\n" + "\n\n".join(parts)
    if rows:
        TABLES["general"] = pd.DataFrame(rows).to_markdown(index=False, disable_numparse=True) + \
            "\n\n: Generalisation: mean test accuracy (%) with other random-kernel transforms, on multivariate UEA datasets and on the largest UCR datasets (seeds 0–2)\n"
        text += "\n\nTABLE:general"
    rows = []
    fam = []
    if "families" in ex:
        f = ex["families"]
        rows += _rows(f"{f['n_datasets']} UCR datasets", f["mean_accuracy"], ["ridge_std", "ridge_wide", "loo_em_bic", "quant", "nn_ed"])
        a = f["mean_accuracy"]
        g = lambda m, c: f"{100 * a[c][m]:.1f}"
        fam.append(f"*Other classifier families* (TABREF:families; {f['n_datasets']} datasets, seeds 0–2). QUANT, whose extremely randomised trees average over many weak learners, loses "
                   f"{100 * (a['sym 0%']['quant'] - a['sym 40%']['quant']):.1f} points between clean labels and 40% symmetric noise ({g('quant', 'sym 0%')}% to {g('quant', 'sym 40%')}%), compared with "
                   f"{100 * (a['sym 0%']['ridge_std'] - a['sym 40%']['ridge_std']):.1f} points for default-grid MiniRocket and {100 * (a['sym 0%']['ridge_wide'] - a['sym 40%']['ridge_wide']):.1f} for the wide grid. "
                   f"At 40% asymmetric noise QUANT reaches {g('quant', 'asym 40%')}%, wide-grid ridge {g('ridge_wide', 'asym 40%')}% and LOO-EM {g('loo_em_bic', 'asym 40%')}% "
                   f"(LOO-EM versus QUANT: {_d(ctx, 'families', 'loo_em_bic', 'quant', 'asym 40%')}; at 40% symmetric noise {_d(ctx, 'families', 'loo_em_bic', 'quant', 'sym 40%')}). "
                   f"The 1-nearest-neighbour classifier is the most fragile method tested ({g('nn_ed', 'sym 0%')}% clean, {g('nn_ed', 'sym 40%')}% at 40% symmetric noise).")
    if "deep" in ex:
        f = ex["deep"]
        dc = [c for c in ("sym 0%", "sym 30%", "asym 30%") if c in f["mean_accuracy"]]
        r2 = _rows(f"{f['n_datasets']} small UCR datasets", f["mean_accuracy"], ["ridge_std", "ridge_wide", "loo_em_bic", "fcn_ce", "fcn_gce", "fcn_coteach"], dc)
        TABLES["deep"] = pd.DataFrame(r2).to_markdown(index=False, disable_numparse=True) + \
            f"\n\n: Deep baselines on {f['n_datasets']} small UCR datasets (seeds 0–2): mean test accuracy (%)\n"
        a = f["mean_accuracy"]
        g = lambda m, c: f"{100 * a[c][m]:.1f}"
        t = f["mean_time_s"]
        fam.append(f"*Deep baselines* (TABREF:deep; {f['n_datasets']} small datasets, seeds 0–2, CPU training). At 30% symmetric noise a fully convolutional network reaches "
                   f"{g('fcn_ce', 'sym 30%')}% with cross-entropy, {g('fcn_gce', 'sym 30%')}% with the generalised cross-entropy and {g('fcn_coteach', 'sym 30%')}% with Co-teaching given the true noise rate; "
                   f"wide-grid ridge reaches {g('ridge_wide', 'sym 30%')}% and LOO-EM {g('loo_em_bic', 'sym 30%')}%. At 30% asymmetric noise the figures are {g('fcn_ce', 'asym 30%')}%, {g('fcn_gce', 'asym 30%')}%, "
                   f"{g('fcn_coteach', 'asym 30%')}%, {g('ridge_wide', 'asym 30%')}% and {g('loo_em_bic', 'asym 30%')}%; on clean labels {g('fcn_ce', 'sym 0%')}%, {g('fcn_gce', 'sym 0%')}%, {g('fcn_coteach', 'sym 0%')}%, "
                   f"{g('ridge_wide', 'sym 0%')}% and {g('loo_em_bic', 'sym 0%')}%. One network takes {t.get('fcn_ce', float('nan')):.0f} s on average on one CPU core "
                   f"(Co-teaching {t.get('fcn_coteach', float('nan')):.0f} s). On these datasets Co-teaching, which is told the true noise rate, is the most accurate method at 30% symmetric noise, and the ridge variants are the most accurate on clean labels and under asymmetric noise. "
                   f"The comparison covers few, small datasets and simple networks and supports no general claim about deep noisy-label methods.")
    if rows:
        TABLES["families"] = pd.DataFrame(rows).to_markdown(index=False, disable_numparse=True) + \
            "\n\n: Other classifier families trained on the same noisy labels (seeds 0–2): mean test accuracy (%)\n"
    text += "\n\n" + "\n\n".join(fam)
    if rows:
        text += "\n\nTABLE:families"
    if "deep" in TABLES:
        text += "\n\nTABLE:deep"
    sec["generalisation"] = text

    # ------------------------------------------------------------------ cost
    e = ctx.metrics["efficiency"]
    b = ctx.metrics["base_cost"]
    f3 = lambda v: f"{v:.3f}"
    cost_rows = [{"Method": ag.LABEL[m], "Median time (s)": f3(e["median_time_s"][m]), "Mean time (s)": f3(e["mean_time_s"][m]),
                  "Median ratio to eigendecomposition": f"{e['time_over_eig_median'][m]:.2f}"}
                 for m in ("ridge_std", "ridge_wide", "loo_em_bic", "loo_em", "conf_learning_cal", "kfold_relabel", "enn", "loo_relabel", "loo_remove")]
    cap = "Wall-clock cost per run of each method beyond the shared feature transform and eigendecomposition (single CPU thread, all datasets and conditions)"
    key = "cost"
    TABLES[key] = pd.DataFrame(cost_rows).to_markdown(index=False, disable_numparse=True) + f"\n\n: {cap}\n"
    text = (f"TABREF:cost and Fig. 9 report the time each method adds to the shared work, which is the MiniRocket transform (median {b['median_t_feat']:.2f} s per dataset) and one eigendecomposition "
            f"(median {b['median_t_eig']:.3f} s). Selecting the penalty on the wide instead of the default grid costs {1000 * (e['median_time_s']['ridge_wide'] - e['median_time_s']['ridge_std']):.0f} ms at the median. "
            f"LOO-EM takes a median of {e['median_time_s']['loo_em_bic']:.2f} s (mean {e['mean_time_s']['loo_em_bic']:.2f} s), {e['time_over_eig_median']['loo_em_bic']:.1f} times the eigendecomposition; "
            f"5-fold confident learning takes {e['median_time_s']['conf_learning_cal']:.2f} s (mean {e['mean_time_s']['conf_learning_cal']:.2f} s), {e['time_over_eig_median']['conf_learning_cal']:.1f} times. "
            f"LOO-EM is therefore not free, as a single LOO relabelling round is ({e['median_time_s']['loo_relabel'] * 1000:.0f} ms), but it is about "
            f"{e['mean_time_s']['conf_learning_cal'] / e['mean_time_s']['loo_em_bic']:.0f} times cheaper on average than the cross-validated alternative and never refits a model. "
            f"Memory is dominated by the $n\\times n$ eigenvector matrix and its element-wise square. All ridge variants deploy the same linear classifier, so model size and inference cost are identical; energy use was not measured.")
    if "large" in ex:
        tm = pd.DataFrame(ex["large"]["timing"])
        big = tm[tm.n_train == tm.n_train.max()]
        if len(big):
            r = big.set_index("method")
            text += (f" On the largest dataset processed ({big.dataset.iloc[0]}, {int(big.n_train.iloc[0]):,} training series, two BLAS threads) the eigendecomposition takes "
                     f"{r.t_eig.iloc[0]:.0f} s and LOO-EM {r.time_s['loo_em_bic']:.0f} s per noise condition.")
    text += f"\n\nTABLE:{key}"
    sec["cost"] = text

    # ------------------------------------------------------------------ sensitivity
    s = pd.DataFrame(ex.get("sensitivity", []))
    if len(s):
        def v(tr, m, c):
            r = s[(s["transform"] == tr) & (s.method == m)]
            return float("nan") if r.empty else 100 * float(r[c].iloc[0])
        base = "minirocket"
        def line(m, label):
            return f"{label}: {v(base, m, 'sym 0%'):.1f} / {v(base, m, 'sym 40%'):.1f} / {v(base, m, 'asym 40%'):.1f}"
        kern = "; ".join(f"{tr.split('@')[1] if '@' in tr else '10,000'} kernels: ridge {v(tr, 'ridge_wide', 'sym 40%'):.1f}, LOO-EM {v(tr, 'loo_em_bic', 'sym 40%'):.1f}"
                         for tr in ("minirocket@1000", "minirocket@2500", "minirocket@5000", "minirocket") if (s["transform"] == tr).any())
        n_s = 30
        text = (f"Sensitivity was examined on a fixed random subset of {n_s} datasets outside the development set (seeds 0–2; full table in the supplement). "
                f"Mean accuracy (%) on clean labels / at 40% symmetric / at 40% asymmetric noise is, for the reference setting, {line('loo_em_bic', 'LOO-EM')}; "
                f"{line('bic_pen_0.5', 'BIC penalty ×0.5')}; {line('bic_pen_2', 'BIC penalty ×2')}; {line('bic_pen_4', 'BIC penalty ×4')}; "
                f"{line('bic_prior_1', 'prior strength 1')}; {line('bic_prior_50', 'prior strength 50')}; {line('bic_iter_5', '5 EM iterations')}; {line('bic_iter_50', '50 EM iterations')}; "
                f"{line('ridge_wide', 'wide-grid ridge')}. The BIC penalty acts as described in Sect. 6.4: a smaller penalty buys accuracy under asymmetric noise "
                f"with accuracy on clean labels. A strong prior on the transition matrix (50 pseudo-counts per class) pulls the full model towards the symmetric one and lowers accuracy under asymmetric noise; "
                f"the number of EM iterations is immaterial within the range tried. "
                f"Varying the number of kernels at 40% symmetric noise gives {kern}; the ordering of the methods does not depend on it.")
    else:
        text = "Sensitivity results are not available in this build."
    sec["sensitivity"] = text
    return sec
