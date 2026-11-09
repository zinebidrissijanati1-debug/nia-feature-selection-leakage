"""Checks that the study environment works on this computer. Run:  python check_env.py"""
import os, sys, platform, time
print("Python  :", sys.version.split()[0], "| OS:", platform.platform(), "| CPU cores:", os.cpu_count())
if sys.version_info[:2] < (3, 11):
    print("WARNING: Python 3.11 or newer is recommended (tested on 3.13).")

import numpy, scipy, pandas, sklearn, joblib, statsmodels, matplotlib
for m in (numpy, scipy, pandas, sklearn, joblib, statsmodels, matplotlib):
    print(f"  {m.__name__:12s} {m.__version__}")

# 1) UCI download (Wisconsin breast cancer, UCI id 17)
from ucimlrepo import fetch_ucirepo
d = fetch_ucirepo(id=17)
print("UCI 17 (WDBC) downloaded:", d.data.features.shape, "-> expected (569, 30)")

# 2) OpenML download (Leukemia, OpenML id 1104)
import openml
openml.config.cache_directory = os.path.join(os.getcwd(), "openml_cache")
ds = openml.datasets.get_dataset(1104, download_data=True)
X, y, _, _ = ds.get_data(target=ds.default_target_attribute)
print("OpenML 1104 (Leukemia) downloaded:", X.shape, "-> expected (72, 7129)")

# 3) Speed test: one KNN fitness evaluation (5-fold CV)
from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
Xb, yb = load_breast_cancer(return_X_y=True)
model = make_pipeline(StandardScaler(), KNeighborsClassifier(5))
t = time.time()
for _ in range(20):
    cross_val_predict(model, Xb, yb, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")
per_eval = (time.time() - t) / 20
hours = 59 * per_eval / 0.02 / max(1, (os.cpu_count() or 1) - 1)
print(f"One fitness evaluation: {per_eval:.3f} s  ->  whole study roughly {hours:.0f} h using {max(1,(os.cpu_count() or 1)-1)} cores")
print("\nAll checks passed.")
