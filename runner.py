"""ONE runner for every experiment and dataset of the study.

Examples
    python runner.py --dataset D01                                  # all runs of D01
    python runner.py --dataset D01 D11 --algorithm GA ALL_FEATURES --seed 1 --experiment E1_protocol E1_baselines E2_permuted_labels
    python runner.py --dataset D01 --jobs 7                         # 7 runs in parallel

Each run (one row of run_matrix.csv) writes results/<run_id>.json. Finished runs are skipped,
so the script can be stopped and restarted at any time.

Design (see experiment_design.md):
  protocols  non_nested               : NIA fitness and reported score use the SAME 5 folds of all data
             non_nested_80pct_control : non_nested on a random 80% subsample (matches the selection data size of nested CV)
             nested_5x5               : outer 5 folds held out; NIA searches with inner 5-fold CV on the outer-train part
  fitness    error_rate : 0.99*(1 - balanced accuracy) + 0.01*(selected/total)
             brier      : 0.99*Brier score              + 0.01*(selected/total)
  every algorithm gets exactly `budget_evals` fitness evaluations (counted centrally)
  initialisation (option B, sparse start): each feature starts selected with probability min(0.5, 50/p)
  preprocessing (median imputation, scaling) is fitted inside training data only, in all protocols
  for one dataset and seed, all methods use the SAME outer folds (paired comparisons)
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1"); os.environ.setdefault("MKL_NUM_THREADS", "1")
import argparse, json, math, time, warnings
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, brier_score_loss, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
POP = 20                      # population size for every NIA
ALPHA = 0.99                  # weight of the error term in the fitness
INIT_FEATURES = 50            # option B (sparse start): each feature starts selected with probability min(0.5, 50/p)


def init_prob(d):
    """Starting probability that a feature is selected. Equals 0.5 for p <= 100 (all tabular datasets);
    smaller for high-dimensional data so the search starts from about 50 features."""
    return min(0.5, INIT_FEATURES / d)


# ----------------------------------------------------------------------------- data and folds
def load(did):
    df = pd.read_csv(os.path.join(HERE, "data", f"{did}.csv"))
    groups = df.pop("group").to_numpy() if "group" in df.columns else None
    y = df.pop("y").to_numpy().astype(int)
    return df.to_numpy(dtype=float), y, groups


def splitter(seed, grouped):
    return (StratifiedGroupKFold(5, shuffle=True, random_state=seed) if grouped
            else StratifiedKFold(5, shuffle=True, random_state=seed))


def folds(y, groups, seed):
    return list(splitter(seed, groups is not None).split(np.zeros(len(y)), y, groups))


# ----------------------------------------------------------------------------- models and metrics
def make_model(name, seed):
    if name == "KNN5":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), KNeighborsClassifier(5))
    if name == "RF100":
        return make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(100, random_state=seed, n_jobs=1))
    raise ValueError(name)


def oof_proba(model, X, y, fold_list, cols):
    """Out-of-fold probability of y = 1 on the given folds, using feature columns `cols`."""
    p = np.full(len(y), np.nan)
    for tr, te in fold_list:
        m = clone(model).fit(X[tr][:, cols], y[tr])
        proba = m.predict_proba(X[te][:, cols])
        p[te] = proba[:, list(m.classes_).index(1)] if 1 in m.classes_ else 0.0
    return p


def metrics(y, p):
    yhat = (p >= 0.5).astype(int)
    out = {"accuracy": accuracy_score(y, yhat), "balanced_accuracy": balanced_accuracy_score(y, yhat),
           "sensitivity": recall_score(y, yhat, pos_label=1, zero_division=0),
           "specificity": recall_score(y, yhat, pos_label=0, zero_division=0),
           "auc": roc_auc_score(y, p) if len(np.unique(y)) == 2 else float("nan"),
           "brier": brier_score_loss(y, p)}
    # calibration slope and intercept: logistic recalibration of y on logit(p)
    pc = np.clip(p, 1e-4, 1 - 1e-4); lo = np.log(pc / (1 - pc)).reshape(-1, 1)
    try:
        lr = LogisticRegression(C=1e6, solver="liblinear", max_iter=1000).fit(lo, y)
        out["cal_slope"], out["cal_intercept"] = float(lr.coef_[0, 0]), float(lr.intercept_[0])
    except Exception:
        out["cal_slope"] = out["cal_intercept"] = float("nan")
    bins = np.minimum((p * 10).astype(int), 9)          # ECE, 10 equal-width bins
    out["ece"] = float(sum(abs(p[bins == b].mean() - y[bins == b].mean()) * (bins == b).mean() for b in range(10) if (bins == b).any()))
    return {k: float(v) for k, v in out.items()}


def fitness_from(y, p, k, n_feat, kind):
    if kind == "error_rate":
        err = 1 - balanced_accuracy_score(y, (p >= 0.5).astype(int))
    elif kind == "brier":
        err = brier_score_loss(y, p)
    else:
        raise ValueError(kind)
    return ALPHA * err + (1 - ALPHA) * k / n_feat


class Fitness:
    """Counts evaluations centrally so every algorithm gets exactly the same budget."""
    def __init__(self, model, X, y, fold_list, kind, budget):
        self.model, self.X, self.y, self.folds, self.kind, self.budget = model, X, y, fold_list, kind, budget
        self.used, self.cache, self.best, self.best_mask = 0, {}, math.inf, None

    def __call__(self, mask):
        if not mask.any():
            return 1.0
        self.used += 1
        key = mask.tobytes()
        if key not in self.cache:
            cols = np.flatnonzero(mask)
            p = oof_proba(self.model, self.X, self.y, self.folds, cols)
            self.cache[key] = fitness_from(self.y, p, len(cols), mask.size, self.kind)
        f = self.cache[key]
        if f < self.best:
            self.best, self.best_mask = f, mask.copy()
        return f

    @property
    def done(self):
        return self.used >= self.budget


# ----------------------------------------------------------------------------- nature-inspired algorithms (binary)
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -30, 30)))


def binarize(x, rng):
    m = rng.random(x.shape) < sigmoid(x)
    if not m.any():
        m[rng.integers(m.size)] = True
    return m


def run_ga(F, d, rng):
    pop = rng.random((POP, d)) < init_prob(d)
    for m in pop:
        if not m.any():
            m[rng.integers(d)] = True
    fit = np.array([F(m) for m in pop])
    while not F.done:
        new = [pop[np.argmin(fit)].copy()]                               # elitism (1)
        while len(new) < POP:
            a, b = rng.integers(POP, size=2); p1 = pop[a] if fit[a] < fit[b] else pop[b]
            a, b = rng.integers(POP, size=2); p2 = pop[a] if fit[a] < fit[b] else pop[b]
            child = np.where(rng.random(d) < 0.5, p1, p2) if rng.random() < 0.9 else p1.copy()   # uniform crossover
            flip = rng.random(d) < 1.0 / d                                # bit-flip mutation
            child = child ^ flip
            if not child.any():
                child[rng.integers(d)] = True
            new.append(child)
        pop = np.array(new)
        fit = np.array([F(m) if not F.done else np.inf for m in pop])
    return F.best_mask


def continuous_nia(update):
    """Wrap a continuous update rule (positions in [-6, 6]^d) with S-shaped binarisation."""
    def run(F, d, rng):
        # sparse start: positions whose sigmoid gives a mean selection probability of init_prob(d)
        # (for p <= 100 this is the usual uniform start on [-6, 6])
        p0 = init_prob(d)
        if p0 >= 0.5:
            X = rng.uniform(-6, 6, (POP, d))
        else:
            u = np.clip(rng.uniform(0, 2 * p0, (POP, d)), 1e-6, 1 - 1e-6)
            X = np.clip(np.log(u / (1 - u)), -6, 6)
        masks = [binarize(x, rng) for x in X]
        fit = np.array([F(m) for m in masks])
        state = {"t": 0}
        T = max(1, F.budget // POP)
        while not F.done:
            X = update(X, fit, rng, state, T, F, d)
            X = np.clip(X, -6, 6)
            masks = [binarize(x, rng) for x in X]
            fit = np.array([F(m) if not F.done else np.inf for m in masks])
            state["t"] += 1
        return F.best_mask
    return run


def pso_update(X, fit, rng, s, T, F, d):          # binary PSO (Kennedy & Eberhart 1997 style), w decreasing 0.9 -> 0.4
    if "V" not in s:
        s["V"] = np.zeros_like(X); s["P"] = X.copy(); s["Pf"] = fit.copy()
    imp = fit < s["Pf"]; s["P"][imp] = X[imp]; s["Pf"][imp] = fit[imp]
    g = s["P"][np.argmin(s["Pf"])]
    w = 0.9 - 0.5 * min(1, s["t"] / T)
    s["V"] = np.clip(w * s["V"] + 2 * rng.random(X.shape) * (s["P"] - X) + 2 * rng.random(X.shape) * (g - X), -6, 6)
    return X + s["V"]


def gwo_update(X, fit, rng, s, T, F, d):          # grey wolf optimizer (Mirjalili et al. 2014)
    order = np.argsort(fit); A_, B_, D_ = X[order[0]], X[order[1]], X[order[2]]
    a = 2 - 2 * min(1, s["t"] / T)
    new = np.zeros_like(X)
    for leader in (A_, B_, D_):
        r1, r2 = rng.random(X.shape), rng.random(X.shape)
        A, C = 2 * a * r1 - a, 2 * r2
        new += leader - A * np.abs(C * leader - X)
    return new / 3


def woa_update(X, fit, rng, s, T, F, d):          # whale optimization algorithm (Mirjalili & Lewis 2016)
    best = X[np.argmin(fit)]
    a = 2 - 2 * min(1, s["t"] / T)
    new = np.empty_like(X)
    for i in range(len(X)):
        r = rng.random(); A = 2 * a * rng.random() - a; C = 2 * rng.random()
        if rng.random() < 0.5:
            ref = best if abs(A) < 1 else X[rng.integers(len(X))]
            new[i] = ref - A * np.abs(C * ref - X[i])
        else:
            l = rng.uniform(-1, 1)
            new[i] = np.abs(best - X[i]) * np.exp(l) * np.cos(2 * np.pi * l) + best
    return new


def levy(rng, d, beta=1.5):
    sig = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    return 0.01 * rng.normal(0, sig, d) / np.abs(rng.normal(0, 1, d)) ** (1 / beta)


def hho_update(X, fit, rng, s, T, F, d):          # Harris hawks optimization (Heidari et al. 2019), with greedy dives
    best = X[np.argmin(fit)]; mean = X.mean(axis=0)
    new = X.copy()
    for i in range(len(X)):
        E = 2 * (2 * rng.random() - 1) * (1 - min(1, s["t"] / T))
        if abs(E) >= 1:                                   # exploration
            if rng.random() >= 0.5:
                Xr = X[rng.integers(len(X))]
                new[i] = Xr - rng.random() * np.abs(Xr - 2 * rng.random() * X[i])
            else:
                new[i] = (best - mean) - rng.random() * (-6 + rng.random() * 12)
        else:                                             # exploitation
            r = rng.random(); J = 2 * (1 - rng.random())
            if r >= 0.5 and abs(E) >= 0.5:
                new[i] = (best - X[i]) - E * np.abs(J * best - X[i])
            elif r >= 0.5:
                new[i] = best - E * np.abs(best - X[i])
            else:                                         # progressive rapid dives (extra evaluations, counted in budget)
                ref = X[i] if abs(E) >= 0.5 else mean
                Y = best - E * np.abs(J * best - ref)
                Z = Y + rng.random(d) * levy(rng, d)
                cand = [Y, Z]
                fc = [F(binarize(c, rng)) if not F.done else np.inf for c in cand]
                new[i] = cand[int(np.argmin(fc))] if min(fc) < fit[i] else X[i]
    return new


NIAS = {"GA": run_ga, "PSO": continuous_nia(pso_update), "GWO": continuous_nia(gwo_update),
        "WOA": continuous_nia(woa_update), "HHO": continuous_nia(hho_update)}


# ----------------------------------------------------------------------------- selection procedures
def select(alg, X, y, fold_list, clf, fit_kind, budget, seed):
    """Return (selected column indices, evaluations used) using ONLY the data passed in."""
    d = X.shape[1]
    if alg == "ALL_FEATURES":
        return np.arange(d), 0
    if alg == "MI_FILTER_TOPK":                           # ranking + k chosen by CV on the data passed in
        rank = np.argsort(-mutual_info_classif(SimpleImputer(strategy="median").fit_transform(X), y, random_state=seed))
        best, best_k = math.inf, d
        for k in [k for k in (5, 10, 20, 50, 100, 200) if k < d] + [d]:
            cols = rank[:k]; p = oof_proba(make_model(clf, seed), X, y, fold_list, cols)
            f = 1 - balanced_accuracy_score(y, (p >= 0.5).astype(int))
            if f < best:
                best, best_k = f, k
        return rank[:best_k], 0
    if alg == "L1_LOGISTIC":                              # C chosen by CV on the data passed in; selection = non-zero weights
        best, best_cols = math.inf, np.arange(d)
        for C in (0.01, 0.03, 0.1, 0.3, 1.0, 3.0):
            m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                              LogisticRegression(penalty="l1", C=C, solver="liblinear", max_iter=2000))
            p = oof_proba(m, X, y, fold_list, np.arange(d))
            f = 1 - balanced_accuracy_score(y, (p >= 0.5).astype(int))
            if f < best:
                coef = clone(m).fit(X, y)[-1].coef_.ravel()
                cols = np.flatnonzero(coef != 0)
                best, best_cols = f, cols if len(cols) else np.arange(d)
        return best_cols, 0
    F = Fitness(make_model(clf, seed), X, y, fold_list, fit_kind, budget)
    mask = NIAS[alg](F, d, np.random.default_rng(seed))
    return np.flatnonzero(mask), F.used


def report_model(alg, clf, seed):
    if alg == "L1_LOGISTIC":   # evaluate L1 baseline with its own classifier on the selected features
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(solver="liblinear", max_iter=2000))
    return make_model(clf, seed)


# ----------------------------------------------------------------------------- one run
def run_one(row):
    X, y, groups = load(row.dataset)
    seed = int(row.seed)
    if row.labels == "permuted":                          # null control: permute labels (by subject if grouped)
        rng = np.random.default_rng(10_000 + seed)
        if groups is None:
            y = rng.permutation(y)
        else:
            ug = pd.unique(groups); lab = {g: y[groups == g][0] for g in ug}
            perm = dict(zip(ug, rng.permutation([lab[g] for g in ug]))); y = np.array([perm[g] for g in groups])
    idx = np.arange(len(y))
    if row.protocol == "non_nested_80pct_control":        # random 80% subsample (by group if grouped)
        rng = np.random.default_rng(20_000 + seed)
        if groups is None:
            tr, _ = next(StratifiedKFold(5, shuffle=True, random_state=20_000 + seed).split(X, y))
        else:
            tr, _ = next(StratifiedGroupKFold(5, shuffle=True, random_state=20_000 + seed).split(X, y, groups))
        idx = np.sort(tr)
    Xs, ys, gs = X[idx], y[idx], (groups[idx] if groups is not None else None)
    outer = folds(ys, gs, seed)
    budget = int(row.budget_evals); clf = row.classifier; fk = row.fitness
    t0 = time.time()
    if row.protocol.startswith("non_nested"):
        cols, used = select(row.algorithm, Xs, ys, outer, clf, fk, budget, seed)          # selection sees ALL folds
        p = oof_proba(report_model(row.algorithm, clf, seed), Xs, ys, outer, cols)          # ...and is scored on them
        selected = [cols.tolist()]; evals = [used]
    else:                                                                                  # nested_5x5
        p = np.full(len(ys), np.nan); selected, evals = [], []
        for k, (tr, te) in enumerate(outer):
            inner = folds(ys[tr], gs[tr] if gs is not None else None, seed * 100 + k)
            cols, used = select(row.algorithm, Xs[tr], ys[tr], inner, clf, fk, budget, seed * 100 + k)
            m = clone(report_model(row.algorithm, clf, seed)).fit(Xs[tr][:, cols], ys[tr])
            pr = m.predict_proba(Xs[te][:, cols]); p[te] = pr[:, list(m.classes_).index(1)]
            selected.append(cols.tolist()); evals.append(used)
    res = {"run": row._asdict(), "metrics": metrics(ys, p), "n_selected": [len(c) for c in selected],
           "n_features": int(X.shape[1]), "evals_used": evals, "seconds": round(time.time() - t0, 2),
           "y": ys.tolist(), "proba": [round(float(v), 6) for v in p], "rows": idx.tolist(),
           "selected": selected if X.shape[1] <= 200 else None}
    return res


def safe_run(row, outdir):
    path = os.path.join(outdir, f"{row.run_id}.json")
    if os.path.exists(path):
        return row.run_id, "skipped"
    try:
        res = run_one(row)
    except Exception as e:
        with open(os.path.join(outdir, f"{row.run_id}.error.txt"), "w") as f:
            f.write(repr(e))
        return row.run_id, f"ERROR {e!r}"
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(res, f)
    os.replace(tmp, path)                                  # atomic: a crash never leaves a half-written result
    m = res["metrics"]
    return row.run_id, f"bal_acc={m['balanced_accuracy']:.3f} brier={m['brier']:.3f} ({res['seconds']} s)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", nargs="+", required=True)
    ap.add_argument("--experiment", nargs="+"); ap.add_argument("--algorithm", nargs="+")
    ap.add_argument("--seed", nargs="+", type=int); ap.add_argument("--protocol", nargs="+")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--matrix", default=os.path.join(HERE, "run_matrix.csv"))
    ap.add_argument("--results", default=os.path.join(HERE, "results"))
    a = ap.parse_args()
    rm = pd.read_csv(a.matrix)
    rm = rm[rm.dataset.isin(a.dataset)]
    for col, vals in (("experiment", a.experiment), ("algorithm", a.algorithm), ("seed", a.seed), ("protocol", a.protocol)):
        if vals:
            rm = rm[rm[col].isin(vals)]
    os.makedirs(a.results, exist_ok=True)
    rows = list(rm.itertuples(index=False))
    print(f"{len(rows)} runs selected; results -> {a.results}")
    for rid, msg in Parallel(n_jobs=a.jobs, return_as="generator")(delayed(safe_run)(r, a.results) for r in rows):
        print(rid, msg, flush=True)


if __name__ == "__main__":
    main()
