"""Final analysis for the paper: all preregistered tests + exploratory analyses + figures.

Usage:   python analyze_all.py
Output:  analysis/report.txt        (the full printed report; paste it back)
         analysis/table_*.csv       (tables for the paper / supplement)
         analysis/fig*.png + .pdf   (figures, 300 dpi)
"""
import glob, json, os, sys, warnings
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, spearmanr

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
RES = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results")
OUT = os.path.join(HERE, "analysis"); os.makedirs(OUT, exist_ok=True)
NIAS = ["GA", "PSO", "GWO", "WOA", "HHO"]; BASE = ["ALL_FEATURES", "MI_FILTER_TOPK", "L1_LOGISTIC"]
rng = np.random.default_rng(2026)
_report = open(os.path.join(OUT, "report.txt"), "w", encoding="utf-8")


def say(*a):
    s = " ".join(str(x) for x in a); print(s); _report.write(s + "\n")


def holm(p):
    p = np.asarray(p, float); out = np.full(len(p), np.nan); ok = ~np.isnan(p); q = p[ok]
    order = np.argsort(q); m = len(q); adj = np.empty(m); run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (m - r) * q[i])); adj[i] = run
    out[ok] = adj; return out


def wil(x, alt):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) < 2 or np.allclose(x, 0):
        return np.nan
    return float(wilcoxon(x, alternative=alt).pvalue)


def boot_median_ci(x, B=10000):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) < 2:
        return np.nan, np.nan, np.nan
    meds = np.median(rng.choice(x, (B, len(x)), replace=True), axis=1)
    return float(np.median(x)), float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


# ------------------------------------------------------------------ load all results
rows, probs = [], {}
for f in glob.glob(os.path.join(RES, "R*.json")):
    r = json.load(open(f)); u = r["run"]; m = r["metrics"]
    rows.append({**{k: u[k] for k in ("run_id", "experiment", "dataset", "algorithm", "classifier", "protocol", "fitness", "budget_evals", "seed", "labels")},
                 **m, "n_selected_mean": float(np.mean(r["n_selected"])), "seconds": r["seconds"]})
    probs[u["run_id"]] = (np.array(r["y"]), np.array(r["proba"]))
df = pd.DataFrame(rows); df["fitness"] = df.fitness.astype(str)
df.to_csv(os.path.join(OUT, "table_all_runs.csv"), index=False)
plan = pd.read_csv(os.path.join(HERE, "run_matrix.csv"))
say(f"== COMPLETENESS ==\nplanned {len(plan)} | finished {df.run_id.isin(plan.run_id).sum()} | error files {len(glob.glob(os.path.join(RES, '*.error.txt')))}")
say("per experiment:", df.groupby("experiment").size().to_dict())
say(f"total compute: {df.seconds.sum() / 3600:.1f} job-hours")

man = json.load(open(os.path.join(HERE, "data", "manifest.json")))
props = pd.DataFrame({d: {"n": v["n"], "p": v["p"]} for d, v in man.items()}).T.astype(float)
if "D08" in props.index:
    props.loc["D08", "n"] = 32.0
props["log_p_over_n"] = np.log10(props.p / props.n)
ORDER = list(props.sort_values("log_p_over_n").index)


def delta(sub, metric, a, b, col="protocol"):
    t = sub.pivot_table(index=["dataset", "algorithm", "seed"], columns=col, values=metric)
    if a not in t or b not in t:
        return pd.DataFrame()
    return (t[a] - t[b]).groupby(["dataset", "algorithm"]).mean().unstack()


e1 = df[(df.experiment.isin(["E1_protocol", "E1_baselines"])) & (df.labels == "true")]

# ------------------------------------------------------------------ H1, H2, H3 (+ calibration slope)
say("\n== H1 / H2: optimism Δ = non-nested − nested (KNN, accuracy fitness), one-sided Wilcoxon across datasets, Holm over 5 NIAs ==")
tb = delta(e1, "balanced_accuracy", "non_nested", "nested_5x5"); tr = delta(e1, "brier", "non_nested", "nested_5x5")
ts = delta(e1, "cal_slope", "non_nested", "nested_5x5"); tsn = delta(e1, "sensitivity", "non_nested", "nested_5x5")
ta = delta(e1, "auc", "non_nested", "nested_5x5")
h = []
for a in NIAS:
    if a not in tb:
        continue
    x, y = tb[a].dropna(), tr[a].dropna()
    h.append([a, len(x), 100 * x.median(), int((x > 0).sum()), wil(x, "greater"), y.median(), int((y < 0).sum()), wil(y, "less")])
H = pd.DataFrame(h, columns=["algorithm", "n_datasets", "median_dBalAcc_pp", "n_positive", "p_H1", "median_dBrier", "n_negative", "p_H2"])
H["p_H1_holm"] = holm(H.p_H1); H["p_H2_holm"] = holm(H.p_H2)
say(H.round(4).to_string(index=False)); H.to_csv(os.path.join(OUT, "table_H1_H2.csv"), index=False)
for name, t in (("balanced_accuracy", tb), ("brier", tr), ("sensitivity", tsn), ("auc", ta), ("cal_slope", ts)):
    t.reindex(ORDER).to_csv(os.path.join(OUT, f"table_delta_{name}.csv"))

say("\n== Calibration slope (exploratory): Δ = non-nested − nested; median over datasets; two-sided Wilcoxon ==")
for a in [a for a in NIAS if a in ts]:
    x = ts[a].dropna(); say(f"{a:4s} median Δslope={x.median():+.3f}  (non-nested median slope {e1[(e1.algorithm == a) & (e1.protocol == 'non_nested')].cal_slope.median():.2f} vs nested {e1[(e1.algorithm == a) & (e1.protocol == 'nested_5x5')].cal_slope.median():.2f})  p={wil(x, 'two-sided'):.4f}")

say("\n== H3: Spearman(Δ balanced acc, log10 p/n) ==")
for a in [a for a in NIAS if a in tb]:
    x = tb[a].dropna(); rho, p = spearmanr(props.loc[x.index, "log_p_over_n"], x); say(f"{a:4s} rho={rho:+.2f} p={p:.4f}")
pooled = tb[[a for a in NIAS if a in tb]].mean(axis=1).dropna()
rho, p = spearmanr(props.loc[pooled.index, "log_p_over_n"], pooled); say(f"mean of NIAs rho={rho:+.2f} p={p:.4f}")

say("\n== H3 mixed model: balanced_accuracy ~ nonnested * log_p_over_n + C(algorithm), random intercepts for dataset and dataset:seed ==")
try:
    import statsmodels.formula.api as smf
    mm = e1[e1.algorithm.isin(NIAS) & e1.protocol.isin(["non_nested", "nested_5x5"])].copy()
    mm["nonnested"] = (mm.protocol == "non_nested").astype(int)
    mm["log_p_over_n"] = props.loc[mm.dataset, "log_p_over_n"].values
    mm["ds_seed"] = mm.dataset + "_" + mm.seed.astype(str)
    fit = smf.mixedlm("balanced_accuracy ~ nonnested * log_p_over_n + C(algorithm)", mm, groups=mm["dataset"],
                      vc_formula={"ds_seed": "0 + C(ds_seed)"}).fit(reml=True)
    for term in ("nonnested", "log_p_over_n", "nonnested:log_p_over_n"):
        lo, hi = fit.conf_int().loc[term]
        say(f"{term:24s} coef={fit.params[term]:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  p={fit.pvalues[term]:.4g}")
    say("(interpretation: nonnested:log_p_over_n = extra optimism per 10-fold increase in p/n)")
except Exception as ex:
    say("mixed model failed:", repr(ex))

say("\n== Size control: P2 (non-nested on 80%) − P3 (nested), balanced acc (pp), median over datasets; one-sided Wilcoxon ==")
tc = delta(e1, "balanced_accuracy", "non_nested_80pct_control", "nested_5x5")
for a in [a for a in NIAS if a in tc]:
    x = tc[a].dropna(); say(f"{a:4s} median {100 * x.median():+.2f} pp, positive {int((x > 0).sum())}/{len(x)}, p={wil(x, 'greater'):.4f}")

# ------------------------------------------------------------------ H4 remedy
say("\n== H4 / H4b: remedy within nested CV, KNN: Brier fitness (E4) − accuracy fitness (E1) ==")
nest = df[(df.protocol == "nested_5x5") & (df.classifier == "KNN5") & (df.labels == "true") & (df.budget_evals == 1000) & df.algorithm.isin(NIAS)]
rb = delta(nest, "brier", "brier", "error_rate", col="fitness"); ra = delta(nest, "balanced_accuracy", "brier", "error_rate", col="fitness")
rauc = delta(nest, "auc", "brier", "error_rate", col="fitness"); rsl = delta(nest, "cal_slope", "brier", "error_rate", col="fitness")
h4 = []
for a in [a for a in NIAS if a in rb]:
    x, y = rb[a].dropna(), ra[a].dropna(); med, lo, hi = boot_median_ci(100 * y)
    h4.append([a, len(x), x.median(), int((x < 0).sum()), wil(x, "less"), med, lo, hi, lo > -1.0, 100 * rauc[a].median(), rsl[a].median()])
H4 = pd.DataFrame(h4, columns=["algorithm", "n_datasets", "median_dBrier", "n_lower_brier", "p_H4", "median_dBalAcc_pp", "CI_low", "CI_high", "noninferior_1pp", "median_dAUC_pp", "median_dSlope"])
if len(H4):
    H4["p_H4_holm"] = holm(H4.p_H4)
    say(H4.round(4).to_string(index=False)); H4.to_csv(os.path.join(OUT, "table_H4.csv"), index=False)
    allx = ra[[a for a in NIAS if a in ra]].stack(); med, lo, hi = boot_median_ci(100 * allx)
    say(f"pooled (all dataset×algorithm cells): median ΔBalAcc {med:+.2f} pp [{lo:+.2f}, {hi:+.2f}]; median ΔBrier {rb.stack().median():+.4f}")
    rb.reindex(ORDER).to_csv(os.path.join(OUT, "table_remedy_dbrier.csv")); ra.reindex(ORDER).to_csv(os.path.join(OUT, "table_remedy_dbalacc.csv"))

# ------------------------------------------------------------------ E2 permuted labels
say("\n== E2 permuted labels: balanced accuracy, mean over seeds and GA/PSO ==")
e2 = df[df.experiment == "E2_permuted_labels"]
P = e2.groupby(["dataset", "protocol"]).balanced_accuracy.mean().unstack()
if {"non_nested", "nested_5x5"} <= set(P.columns):
    P = P.reindex([d for d in ORDER if d in P.index]); P["gap_pp"] = 100 * (P.non_nested - P.nested_5x5)
    say((P * [1, 1, 1]).round(3).to_string()); P.to_csv(os.path.join(OUT, "table_E2_permuted.csv"))
    say(f"median non-nested {P.non_nested.median():.3f} | median nested {P.nested_5x5.median():.3f} | non-nested > 0.55 on {int((P.non_nested > 0.55).sum())}/{len(P)} datasets")

# ------------------------------------------------------------------ E5a budget
say("\n== E5a search budget: Δ balanced acc (pp) = non-nested − nested, mean over seeds, GA/PSO (1000 from E1, seeds 1-3) ==")
b = df[(df.classifier == "KNN5") & (df.labels == "true") & (df.fitness == "error_rate") & df.algorithm.isin(["GA", "PSO"])
       & df.dataset.isin(["D01", "D07", "D11", "D13"]) & df.protocol.isin(["non_nested", "nested_5x5"]) & (df.seed <= 3)]
B = b.pivot_table(index=["dataset", "algorithm", "budget_evals", "seed"], columns="protocol", values="balanced_accuracy")
if {"non_nested", "nested_5x5"} <= set(B.columns):
    BB = (100 * (B.non_nested - B.nested_5x5)).groupby(["dataset", "algorithm", "budget_evals"]).mean().unstack()
    say(BB.round(2).to_string()); BB.to_csv(os.path.join(OUT, "table_E5a_budget.csv"))
    if BB.shape[1] >= 2:
        inc = (BB.iloc[:, -1] - BB.iloc[:, 0]); say(f"Δ larger at highest than at lowest budget in {int((inc > 0).sum())}/{len(inc)} dataset×algorithm cells")

# ------------------------------------------------------------------ E5b random forest
say("\n== E5b random forest (seeds 1-3): optimism and remedy ==")
rf = df[df.experiment == "E5b_classifier_RF"]
if len(rf):
    g = rf.groupby(["dataset", "algorithm", "protocol", "fitness"])[["balanced_accuracy", "brier"]].mean().unstack(["protocol", "fitness"])
    out = []
    for (d, a), r in g.iterrows():
        try:
            out.append([d, a, 100 * (r[("balanced_accuracy", "non_nested", "error_rate")] - r[("balanced_accuracy", "nested_5x5", "error_rate")]),
                        r[("brier", "non_nested", "error_rate")] - r[("brier", "nested_5x5", "error_rate")],
                        r[("brier", "nested_5x5", "brier")] - r[("brier", "nested_5x5", "error_rate")],
                        100 * (r[("balanced_accuracy", "nested_5x5", "brier")] - r[("balanced_accuracy", "nested_5x5", "error_rate")])])
        except KeyError:
            pass
    RF = pd.DataFrame(out, columns=["dataset", "algorithm", "optimism_dBalAcc_pp", "optimism_dBrier", "remedy_dBrier", "remedy_dBalAcc_pp"])
    say(RF.round(4).to_string(index=False)); RF.to_csv(os.path.join(OUT, "table_E5b_RF.csv"), index=False)

# ------------------------------------------------------------------ figures
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]           # categorical slots 1-5, fixed order
BLUE, RED, NEUTRAL = "#2a78d6", "#e34948", "#f0efec"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.titlecolor": INK, "axes.titlesize": 10, "figure.facecolor": SURF, "axes.facecolor": SURF,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "legend.frameon": False, "savefig.facecolor": SURF})
NAMES = {"D01": "WDBC", "D02": "Coimbra", "D03": "Cleveland", "D04": "Pima", "D05": "ILPD", "D06": "CKD", "D07": "Hepatitis",
         "D08": "Parkinsons", "D09": "Heart failure", "D10": "ACTG175", "D11": "Leukemia", "D12": "Colon", "D13": "Prostate", "D14": "DLBCL"}


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=300, bbox_inches="tight"); fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight"); plt.close(fig)


# Fig 1: heat map of optimism (balanced accuracy, pp); diverging blue - gray - red, values printed in cells
if not tb.empty:
    cols = [c for c in NIAS + BASE if c in tb.columns]; M = 100 * tb.reindex(ORDER)[cols]
    vmax = max(1.0, np.nanmax(np.abs(M.values)))
    cmap = LinearSegmentedColormap.from_list("div", [BLUE, NEUTRAL, RED])
    fig, ax = plt.subplots(figsize=(6.6, 5.2)); ax.grid(False)
    im = ax.imshow(M.values, cmap=cmap, norm=TwoSlopeNorm(0, -vmax, vmax), aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M.values[i, j]
            if not np.isnan(v):
                v = 0.0 if abs(v) < 0.05 else v
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7, color=("#ffffff" if abs(v) > 0.6 * vmax else INK))
    ax.set_xticks(range(len(cols)), [c.replace("_", " ").replace("MI FILTER TOPK", "MI filter").replace("L1 LOGISTIC", "L1-logistic").replace("ALL FEATURES", "All features") for c in cols], rotation=35, ha="right")
    def fmt_pn(x):
        return f"{x:.0f}" if x >= 10 else (f"{x:.1f}" if x >= 1 else f"{x:.2g}")
    ax.set_yticks(range(len(M)), [f"{d} {NAMES.get(d, '')}  (p/n={fmt_pn(10 ** props.loc[d, 'log_p_over_n'])})" for d in M.index])
    ax.axvline(len([c for c in cols if c in NIAS]) - 0.5, color=SURF, linewidth=3)
    cb = fig.colorbar(im, ax=ax, shrink=0.8); cb.set_label("Optimism Δ balanced accuracy (pp)", color=INK2); cb.outline.set_visible(False)
    ax.set_title("Non-nested minus nested balanced accuracy (mean of 5 seeds)\nrows ordered by features-to-samples ratio", loc="left")
    save(fig, "fig1_optimism_heatmap")

# Fig 2: optimism vs log10(p/n) (one series: mean of 5 NIAs; bar = range across NIAs)
if not tb.empty:
    nia = 100 * tb[[a for a in NIAS if a in tb]]; mean = nia.mean(axis=1); lo, hi = nia.min(axis=1), nia.max(axis=1)
    x = props.loc[mean.index, "log_p_over_n"]
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    ax.vlines(x, lo, hi, color=BLUE, alpha=0.35, linewidth=2)
    ax.scatter(x, mean, s=42, color=BLUE, edgecolor=SURF, linewidth=1.5, zorder=3)
    OFF = {"D04": (6, 9, "left"), "D10": (6, -12, "left"), "D01": (-6, -2, "right"), "D06": (6, -7, "left"), "D09": (45, -3, "left", "arrow"),
           "D12": (-6, 6, "right"), "D14": (0, 4, "center", "top"), "D11": (6, 0, "left"), "D13": (6, 0, "left")}
    for d in mean.index:
        o = OFF.get(d, (5, 3, "left")); dx, dy, ha = o[:3]
        yy = hi[d] if len(o) > 3 and o[3] == "top" else mean[d]          # "top": label sits above the range bar
        arrow = dict(arrowstyle="-", color=MUTED, linewidth=0.6, shrinkA=0, shrinkB=4) if len(o) > 3 and o[3] == "arrow" else None
        ax.annotate(NAMES.get(d, d), (x[d], yy), xytext=(dx, dy), textcoords="offset points", fontsize=7, color=INK2, ha=ha, va="center" if arrow else "baseline", arrowprops=arrow)
    rho, p = spearmanr(x, mean)
    ax.set_xlabel("log10(features / samples)"); ax.set_ylabel("Optimism Δ balanced accuracy (pp)")
    ax.axhline(0, color=AXIS, linewidth=1)
    ax.set_title(f"Optimism grows with dimensionality (mean of 5 algorithms; bar = range)\nSpearman ρ = {rho:+.2f}, p = {p:.3g}", loc="left")
    save(fig, "fig2_optimism_vs_dimensionality")

# Fig 3: permuted labels - dumbbell non-nested vs nested
if "P" in dir() and {"non_nested", "nested_5x5"} <= set(P.columns):
    fig, ax = plt.subplots(figsize=(5.6, 4.2)); yy = np.arange(len(P))
    ax.hlines(yy, P.nested_5x5, P.non_nested, color=GRID, linewidth=2)
    ax.scatter(P.nested_5x5, yy, s=40, color=BLUE, edgecolor=SURF, linewidth=1.5, zorder=3, label="Nested (honest)")
    ax.scatter(P.non_nested, yy, s=40, color=RED, edgecolor=SURF, linewidth=1.5, zorder=3, label="Non-nested")
    ax.axvline(0.5, color=MUTED, linewidth=1, linestyle="--"); ax.text(0.5, len(P) - 0.4, " chance", color=MUTED, fontsize=7)
    ax.set_yticks(yy, [f"{d} {NAMES.get(d, '')}" for d in P.index]); ax.set_xlabel("Balanced accuracy with randomly permuted labels")
    ax.legend(loc="lower right"); ax.set_title("Accuracy from nothing: labels shuffled, mean of GA and PSO, 3 seeds", loc="left")
    save(fig, "fig3_permuted_labels")

# Fig 4: reliability curves pooled over NIAs, tabular vs gene datasets
rel = e1[e1.algorithm.isin(NIAS) & e1.protocol.isin(["non_nested", "nested_5x5"])]
if len(rel):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), sharey=True)
    for ax, (title, dsets) in zip(axes, (("Tabular datasets (D01–D10)", [f"D{i:02d}" for i in range(1, 11)]),
                                         ("Gene-expression datasets (D11–D14)", [f"D{i:02d}" for i in range(11, 15)]))):
        for prot, col, lab in (("nested_5x5", BLUE, "Nested"), ("non_nested", RED, "Non-nested")):
            ids = rel[(rel.protocol == prot) & rel.dataset.isin(dsets)].run_id
            if not len(ids):
                continue
            y = np.concatenate([probs[i][0] for i in ids]); pr = np.concatenate([probs[i][1] for i in ids])
            bins = np.minimum((pr * 10).astype(int), 9); xs, ys = [], []
            for k in range(10):
                if (bins == k).sum() >= 20:
                    xs.append(pr[bins == k].mean()); ys.append(y[bins == k].mean())
            ax.plot(xs, ys, color=col, linewidth=2, marker="o", markersize=4, label=lab)
        ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle="--")
        ax.set_title(title, loc="left"); ax.set_xlabel("Mean predicted probability"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    axes[0].set_ylabel("Observed frequency of disease"); axes[0].legend(loc="upper left")
    fig.suptitle("Reliability curves (5 algorithms, 5 seeds pooled)", x=0.02, y=1.06, ha="left", color=INK, fontsize=10)
    save(fig, "fig4_reliability")

# Fig 5: remedy - each point = dataset × algorithm
if len(H4):
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    for k, a in enumerate([a for a in NIAS if a in rb]):
        ax.scatter(100 * ra[a], rb[a], s=30, color=CAT[k], edgecolor=SURF, linewidth=1.2, label=a, zorder=3)
    ax.axhline(0, color=AXIS, linewidth=1); ax.axvline(0, color=AXIS, linewidth=1); ax.axvline(-1, color=MUTED, linewidth=1, linestyle="--")
    ax.text(-1, ax.get_ylim()[1], " −1 pp margin", color=MUTED, fontsize=7, va="top")
    ax.set_xlabel("Δ nested balanced accuracy, Brier − accuracy fitness (pp)"); ax.set_ylabel("Δ nested Brier score (lower = better)")
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.17)); ax.set_title("Calibration-aware fitness inside nested CV (each point: dataset × algorithm)", loc="left")
    save(fig, "fig5_remedy")

# Fig 6: optimism vs search budget
if "BB" in dir() and BB.shape[1] >= 2:
    fig, ax = plt.subplots(figsize=(5.0, 3.6)); avg = BB.groupby(level="dataset").mean()
    for k, d in enumerate(avg.index):
        ax.plot(avg.columns, avg.loc[d], color=CAT[k], linewidth=2, marker="o", markersize=5, label=f"{d} {NAMES.get(d, '')}")
        ax.annotate(NAMES.get(d, d), (avg.columns[-1], avg.loc[d].iloc[-1]), xytext=(5, 0), textcoords="offset points", fontsize=7, color=INK2, va="center")
    ax.set_xscale("log"); ax.set_xticks(list(avg.columns), [str(c) for c in avg.columns]); ax.minorticks_off()
    ax.set_xlabel("Search budget (fitness evaluations)"); ax.set_ylabel("Optimism Δ balanced accuracy (pp)")
    ax.legend(loc="upper left", fontsize=7); ax.set_title("Optimism vs search budget (mean of GA and PSO, 3 seeds)\nincreases on high-dimensional data, flat on WDBC", loc="left")
    save(fig, "fig6_budget")

say(f"\nTables and figures written to: {OUT}")
_report.close()
