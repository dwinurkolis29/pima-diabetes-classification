# Comparative Evaluation of Tree-Based, Ensemble, and Margin Classifiers for Early Diabetes Mellitus Prediction Under Zero-Leakage Preprocessing and Class Imbalance

**Author:** Mokhammad Dwi Nurkolis  
**Program:** Master of Informatics, Faculty of Industrial Technology, Universitas Islam Indonesia  
**Contact:** `halo@nurkolis.xyz`

---

## Overview

This repository provides a reproducible experimental machine learning benchmark on the **Pima Indians Diabetes Database** (OpenML ID 37 / NIDDK, $N = 768$, $d = 8$ clinical attributes). It addresses three frequent methodological flaws in clinical tabular classification studies:

1. **Biologically Impossible Zero Values (Disguised Missing Data):** Recodes physiological zeroes (`0`) in `Insulin` ($48.70\%$), `SkinThickness` ($29.56\%$), `BloodPressure` ($4.56\%$), `BMI` ($1.43\%$), and `Glucose` ($0.65\%$) as `NaN` prior to median imputation.
2. **Strict Zero-Leakage Preprocessing & Resampling Pipeline:** Encapsulates zero-to-NaN recoding, median imputation (`SimpleImputer`), feature standardization (`StandardScaler`), and minority oversampling (`SMOTE`) inside an `imblearn.pipeline.Pipeline` so every transformation is fitted **exclusively on active training folds** during 5-fold stratified cross-validation.
3. **Three-Stage Experimental Progression & Bias–Variance Analysis:** Evaluates **Decision Tree (CART)**, **Random Forest**, **XGBoost**, and **Support Vector Machine (RBF Kernel)** across:
   - **Experiment 1 (Baseline):** Default hyperparameters, no class-imbalance resampling.
   - **Experiment 2 (Fold-Internal SMOTE):** Fold-isolated SMOTE ($1:1$) with default classifier hyperparameters.
   - **Experiment 3 (Hyperparameter Tuning + SMOTE):** Joint 5-fold stratified `GridSearchCV` optimization of structural regularization and resampling parameters.

---

## Experimental Architecture & Repository Structure

```text
pima-diabetes-classification/
├── README.md                                  # Project documentation & empirical summary
├── requirements.txt                           # Python dependencies
├── run_all.py                                 # Single entrypoint to run EDA, 3-stage experiments, and plots
├── build_deliverables.py                      # Compiles IEEE LaTeX PDF, executed Colab Notebook, and Slide Deck PDF
│
├── features/                                  # Feature-first implementation modules
│   ├── dataset/
│   │   ├── loader.py                          # OpenML ID 37 downloader, schema validator, 80:20 stratified split
│   │   └── eda.py                             # Biological zero audit & publication EDA figures
│   ├── experiments/
│   │   ├── pipelines.py                       # Zero-leakage BiologicalZeroToNanTransformer & imblearn Pipeline
│   │   └── runner.py                          # 3-stage experimental benchmark across DT, RF, XGBoost, and SVM
│   └── evaluation/
│       ├── metrics.py                         # Acc, Prec, Sensitivity, Specificity, F1, G-Mean, ROC/PR-AUC, 95% CI
│       └── plots.py                           # 300 DPI PNG & vector PDF figures (CM, ROC/PR, Learning Curves)
│
├── artifacts/                                 # Generated data, quantitative metrics, and figures
│   ├── data/pima_diabetes_raw.csv
│   ├── results/
│   │   ├── eda_feature_audit.csv
│   │   ├── metrics_summary.csv
│   │   └── best_hyperparameters.json
│   └── figures/                               # Publication-ready 300 DPI PNG & vector PDF charts
│
├── paper/                                     # IEEE Journal Article LaTeX source (IEEEtran)
│   ├── main.tex
│   ├── references.bib
│   └── presentation_slides.tex
│
├── presentation/                              # 5-minute presentation slide outline & narration script
│   ├── slide_presentasi.md
│   └── naskah_narasi_5_menit.md
│
└── deliverables/                              # Ready-to-submit deliverables package
    ├── 01_IEEE_Article_Diabetes_Classification.pdf
    ├── 02_Colab_Notebook_Diabetes_Classification.ipynb
    └── 03_Presentation_Slides_Diabetes_Classification.pdf
```

---

## Key Empirical Results (Hold-Out Stratified Test Set, $N = 154$)

| Stage | Algorithm | Train F1 | 5-Fold CV F1 ($\mu \pm \sigma$) | Test Acc | Test Prec | Test Sens (Recall) | Test Spec | Test F1 | Test ROC-AUC | Test PR-AUC | Train–Test Gap ($\Delta F_1$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Exp 1: Baseline** | Decision Tree | 1.0000 | $0.5518 \pm 0.0201$ | 0.6818 | 0.5532 | 0.4815 | 0.7900 | 0.5149 | 0.6357 | 0.4482 | +0.4851 |
| | Random Forest | 1.0000 | $0.6330 \pm 0.0192$ | **0.7727** | **0.7021** | 0.6111 | **0.8600** | 0.6535 | 0.8181 | 0.6989 | +0.3465 |
| | XGBoost | 1.0000 | $0.6117 \pm 0.0183$ | 0.7662 | 0.6957 | 0.5926 | **0.8600** | 0.6400 | 0.8057 | 0.6763 | +0.3600 |
| | SVM (RBF) | 0.7461 | $0.6485 \pm 0.0187$ | 0.7403 | 0.6522 | 0.5556 | 0.8400 | 0.6000 | 0.7964 | 0.6615 | +0.1461 |
| **Exp 2: SMOTE** | Decision Tree | 1.0000 | $0.6144 \pm 0.0100$ | 0.7013 | 0.5690 | 0.6111 | 0.7500 | 0.5893 | 0.6806 | 0.4841 | +0.4107 |
| | Random Forest | 1.0000 | $0.6683 \pm 0.0356$ | 0.7532 | 0.6333 | 0.7037 | 0.7800 | 0.6667 | 0.8139 | 0.6912 | +0.3333 |
| | XGBoost | 1.0000 | $0.6063 \pm 0.0277$ | 0.7338 | 0.6140 | 0.6481 | 0.7800 | 0.6306 | 0.8106 | 0.6679 | +0.3694 |
| | SVM (RBF) | 0.7930 | $0.6854 \pm 0.0432$ | 0.7273 | 0.5968 | 0.6852 | 0.7500 | 0.6379 | 0.8112 | 0.6728 | +0.1551 |
| **Exp 3: Tuned+SMOTE** | **Decision Tree** | 0.7214 | $0.6739 \pm 0.0246$ | 0.7662 | 0.6324 | **0.7963** | **0.7500** | **0.7049** | 0.8012 | 0.6440 | **+0.0165** |
| | **Random Forest** | 0.8358 | $0.7112 \pm 0.0154$ | 0.7338 | 0.5890 | **0.7963** | 0.7000 | 0.6772 | 0.8146 | 0.6974 | +0.1587 |
| | **XGBoost** | 0.7531 | **0.7121 $\pm$ 0.0209** | 0.7338 | 0.5915 | 0.7778 | 0.7100 | 0.6720 | **0.8224** | **0.7125** | +0.0811 |
| | **SVM (RBF)** | 0.7043 | $0.6956 \pm 0.0091$ | 0.7273 | 0.5882 | 0.7407 | 0.7200 | 0.6557 | 0.8078 | 0.6937 | +0.0486 |

---

## Quick Start & Reproducibility

```powershell
# 1. Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\pip.exe install -r requirements.txt

# 2. Run the complete experimental benchmark and regenerate all CSVs & 300 DPI figures
.\.venv\Scripts\python.exe run_all.py

# 3. Rebuild the executed Colab notebook (.ipynb) and compile the IEEE LaTeX PDF
.\.venv\Scripts\python.exe build_deliverables.py
```
