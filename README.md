# Optimistic and Overconfident: evaluation leakage in nature-inspired wrapper feature selection

Code and results for the article:

> Idrissi Janati, Z., El Fadili Hakim. *Optimistic and Overconfident: Quantifying Evaluation Leakage in Nature-Inspired Wrapper Feature Selection for Medical Diagnosis.* IEEE Access (submitted).

The study compares **non-nested** evaluation (the cross-validation folds that guide the feature-selection search are also used to report performance) with **nested** evaluation (performance measured only on data the search never saw). It covers 14 public medical datasets, five wrappers (GA, PSO, GWO, WOA, HHO) and three baselines, in 2,156 runs.

## Contents

| File | Purpose |
|---|---|
| `prepare_data.py` | Downloads the 14 datasets from UCI and OpenML and applies the fixed cleaning rules → `data/Dxx.csv` + `data/manifest.json` |
| `runner.py` | Runs any subset of the experiments; one result file per run (`results/Rxxxxx.json`), resumable |
| `run_matrix.csv` | The complete list of the 2,156 runs (experiment, dataset, algorithm, protocol, fitness, budget, classifier, seed, labels) |
| `analyze_all.py` | All statistical tests, tables and figures → `analysis/` |
| `summarize.py` | Quick one-line-per-run overview |
| `check_env.py` | Checks the installation and estimates run time |
| `requirements.txt` | Pinned package versions |
| `environment_versions.txt` | Full `pip freeze` of the environment used |
| `experiment_design.md` | Design, hypotheses and analysis plan |
| `datasets.csv` | Dataset list with sources and sizes |

## Reproducing the results

Tested with Python 3.13 on Windows 11 and Linux (identical results on both).

```bash
conda create -n niafs python=3.13 -y
conda activate niafs
pip install -r requirements.txt
python check_env.py
python prepare_data.py
python runner.py --dataset D01 D02 D03 D04 D05 D06 D07 D08 D09 D10 D11 D12 D13 D14 --jobs 7
python analyze_all.py
```

`runner.py` skips finished runs, so it can be stopped and restarted. The full design took 138.3 CPU-hours on an 8-core desktop computer. A subset can be run with `--experiment`, `--algorithm`, `--seed` and `--protocol`, e.g.:

```bash
python runner.py --dataset D01 D11 --algorithm GA ALL_FEATURES --seed 1 --experiment E1_protocol E1_baselines E2_permuted_labels --jobs 4
```

## Data

The datasets are **not redistributed** here; `prepare_data.py` downloads them from their original sources (UCI Machine Learning Repository: ids 17, 451, 45, 225, 336, 46, 174, 519, 890; OpenML: ids 37, 1104, 45087, 45099, 45088). Please cite the original dataset publications listed in the article. The archived `results/` folder (Zenodo) contains the out-of-fold predicted probabilities and metrics of every run, so all analyses can be reproduced without rerunning the experiments.


