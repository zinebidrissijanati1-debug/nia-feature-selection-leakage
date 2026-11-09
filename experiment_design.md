# Experiment design — IEEE Access study

**Working title:** *Optimistic and overconfident: quantifying evaluation leakage in nature-inspired wrapper feature selection for medical diagnosis, and a calibration-aware nested remedy*

**Files in this folder:**
- `datasets.csv`: the 14 datasets, with sizes checked at their source (UCI API and OpenML API, 2026-11-06).
- `run_matrix.csv`: every run (2,156 rows), each with a `run_id`.

**Status:** design proposal. Fix it, preregister it (OSF), and only then run it.

---

## 1. The question, the unit and the responses

| Item | Definition |
|---|---|
| **Question** | How much does non-nested evaluation inflate the apparent performance and calibration of NIA wrapper feature selection, when is the inflation largest, and does a calibration-aware fitness inside nested CV help? |
| **What is varied ("treatments")** | evaluation protocol, NIA algorithm, fitness function, search budget, classifier |
| **Independent unit for conclusions across the field** | the **dataset** (n = 14). Claims such as "the gap grows with p/n" rest on 14 datasets, not on thousands of folds. |
| **Replicates within a dataset** | seeds (repeated CV with new fold splits and new NIA randomness). Seeds estimate within-dataset variability; they are **not** independent evidence across the field. Averaging folds × seeds and then testing across datasets as if all were independent would be **pseudoreplication**. |
| **Responses** | Discrimination: balanced accuracy (primary), accuracy, sensitivity (positive = disease), specificity, AUC. Calibration: **Brier score** (primary), calibration slope and intercept, reliability curve, ECE (secondary). Also: number of selected features, run time. |
| **Primary derived outcome** | **Optimism Δ = metric(non-nested) − metric(nested)**, per dataset × algorithm, averaged over seeds |

## 2. Nuisance factors and how each is controlled

| Nuisance factor | Control |
|---|---|
| Fold split (which patients land in which fold) | **Common random numbers:** for a given dataset and seed, *every* method and protocol uses the **same outer folds**. Comparisons are then paired, which removes split-to-split noise. |
| NIA randomness | Seeded RNG per run; the seed is recorded in `run_matrix.csv` |
| Class imbalance | Stratified folds everywhere |
| Patients with several rows (Parkinson's D08: 195 recordings from 31 people) | **Group folds by subject** (StratifiedGroupKFold). Otherwise the same person appears in train and test, a second leak. |
| Preprocessing leakage (imputation, scaling, encoding) | Every preprocessing step is fitted **inside the training part only**, in *all* protocols, so the only difference between protocols is the feature-selection leakage (see Vrba et al. 2026 for scaling leakage) |
| Unequal search effort between algorithms | Same population size, same number of fitness evaluations (1,000 by default), same transfer function family, same fitness |
| Hidden missing values (Pima zeros in glucose/BP/skin/insulin/BMI; ILPD A/G ratio) | Recode them to missing; impute inside folds; document per dataset |
| Duplicate rows | Detect exact duplicates before the runs; report and remove or group them |
| Hardware time drift | Not relevant to results (computation is deterministic given the seed); record the machine and library versions |

## 3. Datasets (14; sizes checked at source)

| ID | Dataset | Source (id) | n | p | p/n | Notes |
|---|---|---|---|---|---|---|
| D01 | Breast Cancer Wisconsin (Diagnostic) | UCI 17 | 569 | 30 | 0.05 | positive = malignant (label 0 in scikit-learn's copy, so recode) |
| D02 | Breast Cancer Coimbra | UCI 451 | 116 | 9 | 0.08 | small n |
| D03 | Heart Disease (Cleveland) | UCI 45 | 303 | 13 | 0.04 | binarise num > 0; has missing values |
| D04 | Pima Indians Diabetes | OpenML 37 | 768 | 8 | 0.01 | zeros mean missing |
| D05 | ILPD (liver) | UCI 225 | 583 | 10 | 0.02 | check the A/G ratio missing values |
| D06 | Chronic Kidney Disease | UCI 336 | 400 | 24 | 0.06 | many missing values; mixed types |
| D07 | Hepatitis | UCI 46 | 155 | 19 | 0.12 | small n, missing values |
| D08 | Parkinsons (voice) | UCI 174 | 195 recordings / 31 subjects | 22 | 0.71 per subject | **group by subject** |
| D09 | Heart Failure Clinical Records | UCI 519 | 299 | 12 | 0.04 | |
| D10 | AIDS Clinical Trials Group 175 | UCI 890 | 2,139 | 23 | 0.01 | larger-n contrast (gap expected ≈ 0) |
| D11 | Leukemia (ALL/AML) | OpenML 1104 | 72 | 7,129 | 99 | high-dimensional |
| D12 | Colon | OpenML 45087 | 62 | 2,000 | 32 | high-dimensional |
| D13 | Prostate | OpenML 45099 | 102 | 12,600 | 124 | high-dimensional |
| D14 | DLBCL | OpenML 45088 | 77 | 5,469 | 71 | high-dimensional; fix the positive-class coding |

UCI's API reports 197 rows for D08; its own description says 195 recordings from 31 people. Verify after download.

The p/n range spans **0.01 to 124** (four orders of magnitude). That spread is what lets RQ3 test whether the gap grows with dimensionality.

Inclusion rule (write it down before running):
- public;
- binary medical diagnosis or prognosis;
- tabular;
- ≥ 60 samples;
- used in NIA feature-selection literature.

Exclusion: datasets needing images or text. Any dataset added later is reported as a deviation.

## 4. Factors and levels

| Factor | Levels |
|---|---|
| **Protocol** | **P1 non-nested:** the NIA's fitness and the reported score use the same 5 stratified folds of all the data (the common practice). **P2 size control:** P1 run on a random 80% subsample, so the selection fits see the same amount of data as in nested CV; it separates "less data" from "leakage". **P3 nested 5×5:** outer 5 folds are held out; the NIA runs with inner 5-fold CV on the outer-training part only. |
| **Algorithm** | GA, PSO, GWO, WOA, HHO (binary versions with a sigmoid transfer function). **Baselines:** all features (**negative control**: no selection, so Δ should be ≈ 0), mutual-information filter with k tuned inside, L1-logistic. |
| **Fitness** | **F1 accuracy-based:** 0.99 × (1 − balanced accuracy) + 0.01 × (selected/total), a common form in this literature. **F2 calibration-aware:** 0.99 × Brier score + 0.01 × (selected/total). The Brier score is a *proper scoring rule*: it rewards correct ranking and honest probabilities together. |
| **Classifier inside the wrapper** | KNN (k = 5, scaled), the most common choice in this literature; **RF (100 trees)** as a robustness check on a subset |
| **Search budget** | 1,000 fitness evaluations (default); 250 and 4,000 on a subset |
| **Labels** | true; **permuted** (null control) |
| **Seed (replicate)** | 1–5 (main); 1–3 (secondary experiments) |

**Initialisation (decision: option B, sparse start):** each feature starts selected with probability min(0.5, 50/p). For the tabular datasets (p ≤ 100) this is the usual 0.5; for the gene-expression datasets the search starts from about 50 features. Applied identically to all algorithms.

Common settings: population 20, iterations 50 (= 1,000 evaluations), stratified folds, positive class = disease, decision threshold 0.5. Any threshold tuning would happen inside the training data only.

## 5. The experiments

| Exp. | What it tests | Design (runs in `run_matrix.csv`) | Key prediction |
|---|---|---|---|
| **E0 Data audit** | Duplicates, hidden missing values, subject grouping, class balance | per dataset, before anything else | — |
| **E1 Protocol effect (RQ1)** | Size of optimism Δ | 14 datasets × 5 NIAs × 3 protocols × 5 seeds, KNN, F1 (1,050 runs) + baselines × 2 protocols (420) | Δ > 0 for every NIA; Δ ≈ 0 for all-features (negative control); P2 vs P3 still > 0, so leakage, not just data size |
| **E2 Permutation null** | "Accuracy from nothing" | permuted labels, GA and PSO, P1 vs P3, 3 seeds (168) | Nested ≈ 50% balanced accuracy; non-nested **> 50%**, most on high-dimensional data. The clearest demonstration of the problem. |
| **E3 Calibration (RQ2)** | Is the model overconfident? | **no new runs**: computed from E1 predictions | Brier and calibration slope look better under P1 than P3 |
| **E4 Remedy (RQ4)** | Does F2 inside nested CV help? | 14 datasets × 5 NIAs × nested × F2 × 5 seeds (350), compared with E1's nested F1 | Lower nested Brier with F2; nested balanced accuracy not worse by more than 1 point |
| **E5a Search intensity** | Mechanism: more search means more overfitting | 4 datasets (D01, D07, D11, D13) × GA, PSO × budgets 250 / 4,000 × P1, P3 × 3 seeds (96) + E1 budget 1,000 | Δ grows with the search budget |
| **E5b Classifier robustness** | Does the result depend on KNN? | same 4 datasets × GA, PSO × RF100 × (P1-F1, P3-F1, P3-F2) × 3 seeds (72) | Same direction as with KNN |

**Total: 2,156 runs, about 59 single-core hours** (measured cost: ~0.02 s per KNN fitness evaluation and ~0.4 s per RF evaluation, on WDBC-size and 72 × 7,129 data). That is about **15 hours on 4 cores**, of which RF takes about half.

Why a full factorial within each dataset, rather than a fractional design:
- the factors are few, and runs are cheap;
- interactions such as protocol × dimensionality and fitness × algorithm are the questions of interest, so they must not be aliased.

E5 is run on a subset to save time, and is reported as a robustness check.

## 6. Analysis plan (fix before the runs)

**Step 1, within dataset:** for each dataset × algorithm × condition, average over seeds. Report mean ± SD and the paired per-seed differences.

**Step 2, across datasets:** the dataset is the unit.

| Hypothesis | Test | Decision rule |
|---|---|---|
| H1: Δ(balanced accuracy) > 0 | one-sided Wilcoxon signed-rank across 14 datasets, per algorithm; Holm correction over 5 algorithms | Holm-adjusted p < 0.05 |
| H1b: the negative control shows Δ ≈ 0 | same test on all-features | expected not significant (a check that the protocol code is right) |
| H2: Δ(Brier) < 0 (non-nested looks better calibrated) | as for H1 | as for H1 |
| H3: Δ grows with log(p/n) | Spearman correlation across 14 datasets, plus a mixed model `metric ~ protocol × log(p/n) + (1 | dataset) + (1 | dataset:seed)` | 14 datasets give **limited power**; report the effect and its CI and call it confirmatory only if pre-specified |
| H4 (remedy): F2 lowers nested Brier | one-sided Wilcoxon across datasets | p < 0.05 |
| H4b: F2 is not worse in nested balanced accuracy | non-inferiority, margin 1 percentage point (TOST or bootstrap CI of the median difference) | CI lower bound > −1 pp |
| Algorithm comparison (secondary) | Friedman + Nemenyi or Holm post-hoc across datasets | exploratory |
| Permutation null (E2) | per dataset, the distribution of balanced accuracy vs 0.5 | descriptive, with a figure |

**Reported figures:**
1. Δ per dataset (rows ordered by p/n) × algorithm, as a heat map.
2. Δ vs log(p/n), as a scatter plot with a fit.
3. Reliability curves, P1 vs P3.
4. The permutation-null distributions.
5. The F1 vs F2 trade-off.
6. Δ vs search budget.

**Reporting standards:**
- CIs on every difference;
- the number of selected features;
- run times;
- all code and data with a DOI;
- deviations from the plan listed;
- a TRIPOD+AI-style checklist where it applies.

## 7. Threats to validity, and the design answer to each

| Threat | Answer |
|---|---|
| The gap is due to less training data, not leakage | P2 size control |
| A protocol coding bug creates fake gaps | All-features negative control (Δ should be ≈ 0) and the permutation null |
| Same patient in train and test | Grouped folds (D08); duplicate audit (E0) |
| Results specific to KNN or to one budget | E5a and E5b |
| Datasets chosen to get the result | Inclusion rule and list fixed and preregistered before the runs |
| The WBCD pilot was already seen | Not reused; WBCD is rerun fresh with these generic wrappers. Mention the pilot only as motivation. |
| Too few datasets for H3 | Stated as a limitation; effect sizes with CIs reported |

## 8. Implementation checklist

1. **Environment:** Python 3.x with scikit-learn (pin the version), numpy and pandas. NIA implementations: either write your own short binary versions (transparent and easy to review) or use a library such as *mealpy*; pin the version and cite it.
2. **Data:** download each dataset from its source id, save the raw files, write `prepare_Dxx.py` per dataset (recoding, positive class, groups), and record a hash of each file.
3. **One runner** reads a row of `run_matrix.csv` and writes one result file per `run_id` (out-of-fold predicted probabilities, selected features, metrics, time). Saving the **probabilities** lets you compute any calibration metric later without rerunning.
4. **Pilot:** run D01 + D11 × GA × all protocols × 1 seed, and check that the all-features Δ ≈ 0 and the permuted nested score ≈ 0.5 before launching everything.
5. **Preregister on OSF:** hypotheses (section 6), dataset list (section 3), factors (section 4), and this file. Then run.
6. **Archive:** code + `run_matrix.csv` + results to GitHub/Zenodo with a DOI before submitting to IEEE Access.
