"""Download and prepare the 14 study datasets in ONE documented, reproducible way.

Usage:   python prepare_data.py            (all datasets)
         python prepare_data.py D01 D11    (only some)

Output:  data/Dxx.csv        numeric features + column 'y' (1 = disease / positive class) [+ 'group']
         data/manifest.json  source id, n, p, positive class, cleaning notes, SHA-256 of each file

Rules applied to every dataset (decided BEFORE any experiment):
  * positive class (y = 1) = disease / adverse outcome
  * no imputation or scaling here; missing values stay NaN and are imputed INSIDE training folds by runner.py
  * categorical yes/no-type variables -> 0/1; one-hot is not needed (all remaining categoricals are binary
    or ordinal codes)
  * variables that leak the outcome (follow-up time, post-baseline measurements) are removed
  * exact duplicate rows are counted and reported (not silently removed)
"""
import hashlib, json, os, sys, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data")
os.makedirs(OUT, exist_ok=True)


def uci(i):
    from ucimlrepo import fetch_ucirepo
    d = fetch_ucirepo(id=i)
    X = d.data.features.copy()
    X.columns = [str(c) for c in X.columns]
    if X.columns.duplicated().any():                       # e.g. Parkinsons: two 'MDVP:Jitter' columns
        seen = {}
        cols = []
        for c in X.columns:
            seen[c] = seen.get(c, 0) + 1
            cols.append(c if seen[c] == 1 else f"{c}_{seen[c]}")
        X.columns = cols
    y = d.data.targets.iloc[:, 0]
    ids = d.data.ids
    return X, y, ids


def openml_ds(did):
    import openml
    openml.config.cache_directory = os.path.join(HERE, "openml_cache")
    ds = openml.datasets.get_dataset(did, download_data=True)
    X, y, _, _ = ds.get_data(target=ds.default_target_attribute)
    return X.copy(), y.astype(str).str.strip()


def to_num(X):
    return X.apply(pd.to_numeric, errors="coerce")


def yesno(X, cols, mapping):
    for c in cols:
        X[c] = X[c].astype(str).str.strip().str.lower().map(mapping)
    return X


def D01():
    X, y, _ = uci(17)
    return X, (y.astype(str).str.strip() == "M").astype(int), None, "positive = malignant (M)", []

def D02():
    X, y, _ = uci(451)
    return X, (y.astype(int) == 2).astype(int), None, "positive = patient (Classification == 2)", []

def D03():
    X, y, _ = uci(45)
    return to_num(X), (y.astype(int) > 0).astype(int), None, "positive = heart disease (num > 0)", ["6 missing values in ca/thal kept as NaN"]

def D04():
    X, y = openml_ds(37)
    notes = []
    for c in ["plas", "pres", "skin", "insu", "mass"]:          # physiologically impossible zeros = missing
        k = int((X[c] == 0).sum()); X.loc[X[c] == 0, c] = np.nan; notes.append(f"{c}: {k} zeros -> NaN")
    return X, (y == "tested_positive").astype(int), None, "positive = diabetes (tested_positive)", notes

def D05():
    X, y, _ = uci(225)
    X["Gender"] = (X["Gender"].astype(str).str.strip() == "Male").astype(int)
    return to_num(X), (y.astype(int) == 1).astype(int), None, "positive = liver patient (Selector == 1)", ["Gender -> Male=1", "A/G Ratio missing values kept as NaN"]

def D06():
    X, y, _ = uci(336)
    X = yesno(X, ["rbc", "pc"], {"normal": 0, "abnormal": 1})
    X = yesno(X, ["pcc", "ba"], {"notpresent": 0, "present": 1})
    X = yesno(X, ["htn", "dm", "cad", "pe", "ane"], {"no": 0, "yes": 1})
    X = yesno(X, ["appet"], {"good": 0, "poor": 1})
    yy = (y.astype(str).str.strip().str.lower() == "ckd").astype(int)
    return to_num(X), yy, None, "positive = ckd", ["categoricals -> 0/1 (abnormal/present/yes/poor = 1)", "many missing values kept as NaN"]

def D07():
    X, y, _ = uci(46)
    return to_num(X), (y.astype(int) == 1).astype(int), None, "positive = died (Class == 1)", ["missing values kept as NaN"]

def D08():
    X, y, ids = uci(174)
    subj = ids.iloc[:, 0].astype(str).str.extract(r"(S\d+)")[0]   # phon_R01_S01_1 -> S01
    return X, y.astype(int), subj, "positive = Parkinson's (status == 1)", [f"{subj.nunique()} subjects; folds MUST be grouped by subject", "duplicate column names renamed (_2)"]

def D09():
    X, y, _ = uci(519)
    X = X.drop(columns=["time"])
    return X, y.astype(int), None, "positive = death event", ["'time' (follow-up period) removed: it encodes the outcome"]

def D10():
    X, y, _ = uci(890)
    drop = ["time", "offtrt", "cd420", "cd820"]
    X = X.drop(columns=drop)
    return X, y.astype(int), None, "positive = event (cid == 1)", ["removed 'time' (time to failure/censoring) and post-baseline variables offtrt, cd420, cd820: baseline-only prediction"]

def D11():
    X, y = openml_ds(1104)
    return X, (y == "AML").astype(int), None, "positive = AML", []

def D12():
    X, y = openml_ds(45087)
    return X, (y == "-1").astype(int), None, "positive = tumour (label '-1')", ["VERIFY: '-1' assumed = tumour because counts (40 vs 22) match Alon et al. 1999 (40 tumour, 22 normal)"]

def D13():
    X, y = openml_ds(45099)
    return X, (y == "1").astype(int), None, "positive = tumour (label '1')", ["VERIFY: '1' assumed = tumour because counts (52 vs 50) match Singh et al. 2002 (52 tumour, 50 normal)"]

def D14():
    X, y = openml_ds(45088)
    return X, (y == "1").astype(int), None, "positive = follicular lymphoma (label '1', minority)", ["VERIFY: '0' = DLBCL (58), '1' = FL (19), counts match Shipp et al. 2002"]


SOURCES = {"D01": "UCI 17", "D02": "UCI 451", "D03": "UCI 45", "D04": "OpenML 37", "D05": "UCI 225", "D06": "UCI 336",
           "D07": "UCI 46", "D08": "UCI 174", "D09": "UCI 519", "D10": "UCI 890", "D11": "OpenML 1104",
           "D12": "OpenML 45087", "D13": "OpenML 45099", "D14": "OpenML 45088"}


def main(ids):
    man_path = os.path.join(OUT, "manifest.json")
    manifest = json.load(open(man_path)) if os.path.exists(man_path) else {}
    for did in ids:
        X, y, g, pos, notes = globals()[did]()
        X = to_num(X).astype(float)
        X.columns = [f"f{j:05d}" for j in range(X.shape[1])] if X.shape[1] > 200 else X.columns
        df = X.copy(); df["y"] = y.values
        if g is not None:
            df["group"] = g.values
        dup = int(df.drop(columns=["group"], errors="ignore").duplicated().sum())
        if dup > 0 and g is None:                       # keep identical rows in the SAME fold (no copy in train AND test)
            key = pd.util.hash_pandas_object(df, index=False)
            df["group"] = pd.factorize(key)[0]
            g = df["group"]
            notes = notes + [f"{dup} exact duplicate rows: grouped so copies stay in the same fold"]
        path = os.path.join(OUT, f"{did}.csv")
        df.to_csv(path, index=False)
        sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
        manifest[did] = {"source": SOURCES[did], "n": int(len(df)), "p": int(X.shape[1]), "positives": int(y.sum()),
                         "positive_class": pos, "n_missing_cells": int(X.isna().sum().sum()), "exact_duplicate_rows": dup,
                         "grouped": g is not None, "notes": notes, "sha256": sha}
        print(f"{did}: n={len(df)} p={X.shape[1]} positives={int(y.sum())} missing={manifest[did]['n_missing_cells']} "
              f"duplicates={dup} grouped={g is not None} | {pos}")
        for n_ in notes:
            print("     -", n_)
    json.dump(manifest, open(man_path, "w"), indent=1)
    print("manifest written:", man_path)


if __name__ == "__main__":
    main(sys.argv[1:] or [f"D{i:02d}" for i in range(1, 15)])
