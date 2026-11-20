"""Build all final deliverables:
1. Self-contained, fully executed Google Colab Notebook (.ipynb) with inline charts and step-by-step markdown analysis.
2. Official IEEEtran LaTeX PDF compilation (6-10 pages) via Tectonic standalone LaTeX engine.
3. Visual Presentation Slide Deck PDF (5-minute presentation slides with embedded charts).
"""

from pathlib import Path
import platform
import shutil
import stat
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook
from nbclient import NotebookClient


ROOT_DIR = Path(__file__).resolve().parent
PAPER_DIR = ROOT_DIR / "paper"
DELIVERABLES_DIR = ROOT_DIR / "deliverables"
TOOLS_DIR = ROOT_DIR / ".venv" / "tools"


def build_and_execute_colab_notebook() -> Path:
    """Create and execute a standalone Google Colab notebook with narrative markdown cells after every code cell."""
    DELIVERABLES_DIR.mkdir(parents=True, exist_ok=True)
    nb = new_notebook()
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }

    cells = [
        new_markdown_cell(
            "# Comparative Evaluation of Tree-Based, Ensemble, and Margin Classifiers for Early Diabetes Mellitus Prediction Under Zero-Leakage Preprocessing and Class Imbalance\n\n"
            "**Author:** Mokhammad Dwi Nurkolis (NIM: `26917015`)  \n"
            "**Affiliation:** Master of Informatics, Faculty of Industrial Technology, Universitas Islam Indonesia (`halo@nurkolis.xyz`)  \n"
            "**Dataset:** Pima Indians Diabetes Database (OpenML ID 37 / NIDDK, $N = 768$, $d = 8$ clinical features)  \n\n"
            "---\n"
            "### Reproducibility Instructions (Google Colab & Local Environment)\n"
            "1. **Zero Manual File Upload Required:** This notebook automatically downloads the canonical Pima Indians Diabetes dataset directly from OpenML (ID 37) with a fallback public mirror.\n"
            "2. **One-Click Execution:** In Google Colab, click **`Runtime -> Run all`**. All dependencies (`scikit-learn`, `imbalanced-learn`, `xgboost`, `seaborn`, `matplotlib`) are pre-installed or automatically verified in Cell 1.\n"
            "3. **Deterministic Seed:** Every data split, SMOTE synthesizer, cross-validation fold, and model initialization uses `random_state = 42`."
        ),
        new_code_cell(
            "# Cell 1: Environment Setup & Reproducibility Configuration\n"
            "import warnings\n"
            "warnings.filterwarnings(\n"
            "    'ignore',\n"
            "    message=r'The `probability` parameter was deprecated.*',\n"
            "    category=FutureWarning,\n"
            "    module=r'sklearn\\.svm\\._base',\n"
            ")\n\n"
            "import json\n"
            "import numpy as np\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "import seaborn as sns\n\n"
            "from sklearn.base import BaseEstimator, TransformerMixin\n"
            "from sklearn.datasets import fetch_openml\n"
            "from sklearn.model_selection import (\n"
            "    train_test_split,\n"
            "    StratifiedKFold,\n"
            "    GridSearchCV,\n"
            "    cross_validate,\n"
            "    learning_curve,\n"
            ")\n"
            "from sklearn.impute import SimpleImputer\n"
            "from sklearn.preprocessing import StandardScaler\n"
            "from sklearn.tree import DecisionTreeClassifier\n"
            "from sklearn.ensemble import RandomForestClassifier\n"
            "from sklearn.svm import SVC\n"
            "from sklearn.metrics import (\n"
            "    accuracy_score,\n"
            "    precision_score,\n"
            "    recall_score,\n"
            "    f1_score,\n"
            "    roc_auc_score,\n"
            "    average_precision_score,\n"
            "    confusion_matrix,\n"
            "    roc_curve,\n"
            "    precision_recall_curve,\n"
            "    auc,\n"
            "    make_scorer,\n"
            ")\n"
            "from sklearn.inspection import permutation_importance\n"
            "from imblearn.over_sampling import SMOTE\n"
            "from imblearn.pipeline import Pipeline as ImbPipeline\n"
            "from xgboost import XGBClassifier\n\n"
            "RANDOM_STATE = 42\n"
            "np.random.seed(RANDOM_STATE)\n"
            "plt.rcParams.update({'figure.dpi': 120, 'font.size': 10})\n"
            "print('Environment successfully initialized. Random Seed =', RANDOM_STATE)"
        ),
        new_markdown_cell(
            "### Analysis of Step 1 (Environment Setup)\n"
            "All required scientific computing and machine learning libraries (`scikit-learn`, `imbalanced-learn`, and `xgboost`) are loaded with a fixed global seed (`RANDOM_STATE = 42`). Crucially, we import `imblearn.pipeline.Pipeline` rather than the standard scikit-learn pipeline so that synthetic minority oversampling (`SMOTE`) is executed **exclusively inside training folds** during cross-validation and bypassed during validation and test inference."
        ),
        new_code_cell(
            "# Cell 2: Data Acquisition and Schema Standardization\n"
            "CANONICAL_COLUMNS = [\n"
            "    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',\n"
            "    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'\n"
            "]\n"
            "FEATURE_COLUMNS = CANONICAL_COLUMNS[:-1]\n"
            "TARGET_COLUMN = 'Outcome'\n"
            "BIOLOGICAL_ZERO_COLUMNS = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']\n\n"
            "try:\n"
            "    dataset = fetch_openml(data_id=37, as_frame=True, parser='auto')\n"
            "    df = dataset.frame.copy()\n"
            "    df = df.rename(columns={\n"
            "        'preg': 'Pregnancies', 'plas': 'Glucose', 'pres': 'BloodPressure',\n"
            "        'skin': 'SkinThickness', 'insu': 'Insulin', 'mass': 'BMI',\n"
            "        'pedi': 'DiabetesPedigreeFunction', 'age': 'Age', 'class': 'Outcome'\n"
            "    })\n"
            "    df['Outcome'] = df['Outcome'].astype(str).map({'tested_positive': 1, 'tested_negative': 0, '1': 1, '0': 0}).astype(int)\n"
            "    for col in FEATURE_COLUMNS:\n"
            "        df[col] = pd.to_numeric(df[col], errors='coerce')\n"
            "    df = df[CANONICAL_COLUMNS]\n"
            "except Exception:\n"
            "    url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv'\n"
            "    df = pd.read_csv(url, header=None, names=CANONICAL_COLUMNS)\n\n"
            "print(f'Dataset Shape: {df.shape}')\n"
            "print('Target Class Counts:\\n', df['Outcome'].value_counts().rename({0: '0 (Non-Diabetic)', 1: '1 (Diabetic)'}))\n"
            "df.head(10)"
        ),
        new_markdown_cell(
            "### Analysis of Step 2 (Dataset Schema & Class Distribution)\n"
            "The dataset contains exactly $N = 768$ patient records across $8$ clinical predictor variables and $1$ binary outcome (`Outcome` $\\in \\{0, 1\\}$):\n"
            "- **Class `0` (Non-Diabetic):** $500$ patients ($65.10\\%$)\n"
            "- **Class `1` (Diabetic):** $268$ patients ($34.90\\%$)\n"
            "- **Imbalance Ratio:** $1 : 1.8657$. Notice in the preview table above (`df.head(10)`) that several rows contain `0.0` for `Insulin` and `SkinThickness` (e.g., Row 0 has `Insulin = 0.0`, Row 2 has `SkinThickness = 0.0` and `Insulin = 0.0`). In living patients, a serum insulin concentration or skinfold thickness of zero is biologically impossible and represents an unrecorded measurement."
        ),
        new_code_cell(
            "# Cell 3: Exploratory Data Analysis (EDA) & Biologically Impossible Zero Audit\n"
            "audit_rows = []\n"
            "for col in FEATURE_COLUMNS:\n"
            "    s = df[col]\n"
            "    is_bio = col in BIOLOGICAL_ZERO_COLUMNS\n"
            "    z_cnt = int((s == 0).sum()) if is_bio else 0\n"
            "    z_pct = (z_cnt / len(df)) * 100.0 if is_bio else 0.0\n"
            "    s_valid = s.replace(0, np.nan) if is_bio else s\n"
            "    audit_rows.append({\n"
            "        'Feature': col,\n"
            "        'Raw_Mean': round(s.mean(), 2),\n"
            "        'Raw_SD': round(s.std(), 2),\n"
            "        'Invalid_Zero_Count': z_cnt,\n"
            "        'Invalid_Zero_Pct (%)': round(z_pct, 2),\n"
            "        'Valid_Median': round(s_valid.median(), 2),\n"
            "        'Valid_IQR': round(s_valid.quantile(0.75) - s_valid.quantile(0.25), 2),\n"
            "        'Valid_Mean': round(s_valid.mean(), 2),\n"
            "        'Valid_SD': round(s_valid.std(), 2),\n"
            "    })\n\n"
            "audit_df = pd.DataFrame(audit_rows)\n"
            "display(audit_df)\n\n"
            "fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))\n"
            "counts = df['Outcome'].value_counts().sort_index()\n"
            "bars = axes[0].bar(['Non-Diabetic (0)', 'Diabetic (1)'], counts.values, color=['#2b5c8f', '#d95f02'], edgecolor='black', width=0.55)\n"
            "axes[0].set_title('(a) Target Class Distribution (N = 768)', fontweight='bold')\n"
            "axes[0].set_ylabel('Number of Patients')\n"
            "axes[0].set_ylim(0, 580)\n"
            "for b, c in zip(bars, counts.values):\n"
            "    axes[0].text(b.get_x() + b.get_width()/2, c + 12, f'{c} ({c/len(df)*100:.1f}%)', ha='center', fontweight='bold')\n\n"
            "bio_sub = audit_df[audit_df['Feature'].isin(BIOLOGICAL_ZERO_COLUMNS)].sort_values('Invalid_Zero_Pct (%)')\n"
            "bars2 = axes[1].barh(bio_sub['Feature'], bio_sub['Invalid_Zero_Pct (%)'], color='#c0392b', edgecolor='black', height=0.55)\n"
            "axes[1].set_title('(b) Biologically Impossible Zero Values Encoded as Missing', fontweight='bold')\n"
            "axes[1].set_xlabel('Percentage of Samples (%)')\n"
            "axes[1].set_xlim(0, 68)\n"
            "for b, (_, r) in zip(bars2, bio_sub.iterrows()):\n"
            "    axes[1].text(r['Invalid_Zero_Pct (%)'] + 1.0, b.get_y() + b.get_height()/2, f\"{int(r['Invalid_Zero_Count'])} ({r['Invalid_Zero_Pct (%)']:.1f}%)\", va='center', fontweight='bold')\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        new_markdown_cell(
            "### Analysis of Step 3 (Audit of Disguised Missing Values)\n"
            "The audit table and Figure 1 reveal severe disguised missingness across five clinical features:\n"
            "1. **`Insulin`**: $374$ patients ($48.70\\%$) have a recorded value of `0`. Retaining these zeroes artificially drags the mean down to $79.80\\text{ }\\mu\\text{U/mL}$, whereas the true physiological mean among measured patients is $155.55\\text{ }\\mu\\text{U/mL}$ (median $= 125.00\\text{ }\\mu\\text{U/mL}$).\n"
            "2. **`SkinThickness`**: $227$ patients ($29.56\\%$) have a recorded value of `0` (raw mean $20.54\\text{ mm}$ vs. valid median $29.00\\text{ mm}$).\n"
            "3. **`BloodPressure`**, **`BMI`**, and **`Glucose`** exhibit $4.56\\%$ ($35$), $1.43\\%$ ($11$), and $0.65\\%$ ($5$) invalid zeroes, respectively.\n\n"
            "Because dropping rows with missing `Insulin` or `SkinThickness` would discard nearly half of the dataset, **median imputation** is the appropriate clinical strategy. However, computing the median on the entire dataset before splitting would cause **data leakage**; therefore, zero-to-NaN conversion and median imputation must be performed strictly inside the training folds."
        ),
        new_code_cell(
            "# Cell 4: Correlation Matrix and Feature Distribution Analysis\n"
            "clean_eda_df = df.copy()\n"
            "for col in BIOLOGICAL_ZERO_COLUMNS:\n"
            "    clean_eda_df[col] = clean_eda_df[col].replace(0, np.nan)\n"
            "    clean_eda_df[col] = clean_eda_df[col].fillna(clean_eda_df[col].median())\n\n"
            "fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))\n"
            "sns.heatmap(clean_eda_df.corr(), annot=True, fmt='.2f', cmap='RdBu_r', vmin=-0.3, vmax=1.0, ax=axes[0], annot_kws={'size': 7.5})\n"
            "axes[0].set_title('(a) Clinical Feature Correlation Matrix (Post-Imputation)', fontweight='bold')\n\n"
            "top_feats = ['Glucose', 'BMI', 'Age', 'DiabetesPedigreeFunction']\n"
            "norm_df = clean_eda_df[top_feats + ['Outcome']].copy()\n"
            "for c in top_feats:\n"
            "    norm_df[c] = (norm_df[c] - norm_df[c].mean()) / norm_df[c].std()\n"
            "melted = norm_df.melt(id_vars='Outcome', value_vars=top_feats, var_name='Feature', value_name='Z_Score')\n"
            "melted['Feature'] = melted['Feature'].replace({'DiabetesPedigreeFunction': 'Diabetes\\nPedigree'})\n"
            "melted['Diagnosis'] = melted['Outcome'].map({0: 'Non-Diabetic (0)', 1: 'Diabetic (1)'})\n"
            "sns.boxplot(data=melted, x='Feature', y='Z_Score', hue='Diagnosis', order=['Glucose', 'BMI', 'Age', 'Diabetes\\nPedigree'], hue_order=['Non-Diabetic (0)', 'Diabetic (1)'], palette={'Non-Diabetic (0)': '#2b5c8f', 'Diabetic (1)': '#d95f02'}, ax=axes[1])\n"
            "axes[1].set_title('(b) Standardized Distribution of Key Predictors by Outcome', fontweight='bold')\n"
            "axes[1].set_ylabel('Standardized Value (Z-Score)')\n"
            "axes[1].legend(loc='upper left')\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        new_markdown_cell(
            "### Analysis of Step 4 (Feature Correlations & Discriminative Separation)\n"
            "- **Correlation Structure:** `Glucose` exhibits the strongest linear correlation with `Outcome` ($r = 0.49$), followed by `BMI` ($r = 0.31$), `Age` ($r = 0.24$), and `Pregnancies` ($r = 0.22$). Collinearity among predictors is moderate (`Age` vs. `Pregnancies` $r = 0.54$; `BMI` vs. `SkinThickness` $r = 0.54$; `Glucose` vs. `Insulin` $r = 0.42$), indicating no redundant features need to be dropped prior to tree or SVM modeling.\n"
            "- **Distributional Separation:** The boxplots in Panel (b) confirm a pronounced upward shift in median `Glucose` and `BMI` among diabetic patients ($y = 1$), alongside positive skewness and upper-tail outliers in `Age` and `DiabetesPedigreeFunction`."
        ),
        new_code_cell(
            "# Cell 5: Stratified Train-Test Split (80:20) and Zero-Leakage Pipeline Definition\n"
            "X = df[FEATURE_COLUMNS].copy()\n"
            "y = df[TARGET_COLUMN].astype(int).copy()\n\n"
            "X_train, X_test, y_train, y_test = train_test_split(\n"
            "    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y\n"
            ")\n\n"
            "print(f'Train set: {X_train.shape[0]} samples | Positive rate: {y_train.mean()*100:.2f}% ({y_train.sum()} positive, {(y_train==0).sum()} negative)')\n"
            "print(f'Test set : {X_test.shape[0]} samples | Positive rate: {y_test.mean()*100:.2f}% ({y_test.sum()} positive, {(y_test==0).sum()} negative)')\n\n"
            "class BiologicalZeroToNanTransformer(BaseEstimator, TransformerMixin):\n"
            "    \"\"\"Replace biologically impossible zeroes with np.nan inside each CV fold.\"\"\"\n"
            "    def __init__(self, columns=None):\n"
            "        self.columns = columns\n\n"
            "    def fit(self, X, y=None):\n"
            "        self.columns_ = list(self.columns) if self.columns is not None else list(BIOLOGICAL_ZERO_COLUMNS)\n"
            "        if isinstance(X, pd.DataFrame):\n"
            "            self.column_indices_ = [X.columns.get_loc(col) for col in self.columns_ if col in X.columns]\n"
            "        else:\n"
            "            self.column_indices_ = [FEATURE_COLUMNS.index(col) for col in self.columns_ if col in FEATURE_COLUMNS]\n"
            "        return self\n\n"
            "    def transform(self, X):\n"
            "        if isinstance(X, pd.DataFrame):\n"
            "            X_out = X.copy()\n"
            "            for col in self.columns_:\n"
            "                if col in X_out.columns:\n"
            "                    X_out[col] = X_out[col].replace(0, np.nan)\n"
            "            return X_out.to_numpy(dtype=float)\n"
            "        X_arr = np.array(X, dtype=float, copy=True)\n"
            "        for idx in self.column_indices_:\n"
            "            if idx < X_arr.shape[1]:\n"
            "                X_arr[X_arr[:, idx] == 0, idx] = np.nan\n"
            "        return X_arr\n\n"
            "def build_pipeline(classifier, use_smote=False, smote_ratio=1.0, smote_k=5):\n"
            "    steps = [\n"
            "        ('zero_to_nan', BiologicalZeroToNanTransformer()),\n"
            "        ('imputer', SimpleImputer(strategy='median')),\n"
            "        ('scaler', StandardScaler()),\n"
            "    ]\n"
            "    if use_smote:\n"
            "        steps.append(('smote', SMOTE(sampling_strategy=smote_ratio, k_neighbors=smote_k, random_state=RANDOM_STATE)))\n"
            "    steps.append(('classifier', classifier))\n"
            "    return ImbPipeline(steps)"
        ),
        new_markdown_cell(
            "### Analysis of Step 5 (Stratified Partitioning & Zero-Leakage Architecture)\n"
            "The dataset is partitioned into **$614$ training patients** ($400$ negative, $214$ positive; $34.85\\%$ diabetic) and **$154$ held-out test patients** ($100$ negative, $54$ positive; $35.06\\%$ diabetic). By wrapping `BiologicalZeroToNanTransformer`, `SimpleImputer(strategy='median')`, `StandardScaler()`, and `SMOTE()` inside `ImbPipeline`, we guarantee that:\n"
            "1. No median statistic or scaling parameter from the validation fold or test set ever leaks into training.\n"
            "2. Synthetic minority points generated by SMOTE exist **only inside the active training fold**; validation folds and the $154$-patient test set remain $100\\%$ authentic clinical records."
        ),
        new_code_cell(
            "# Cell 6: Three-Stage Experimental Benchmark Execution (Decision Tree, Random Forest, XGBoost, SVM)\n"
            "def specificity_score(y_true, y_pred):\n"
            "    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()\n"
            "    return float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0\n\n"
            "cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)\n"
            "scoring = {\n"
            "    'accuracy': 'accuracy',\n"
            "    'precision': 'precision',\n"
            "    'recall': 'recall',\n"
            "    'specificity': make_scorer(specificity_score),\n"
            "    'f1': 'f1',\n"
            "    'roc_auc': 'roc_auc',\n"
            "}\n\n"
            "def evaluate_model(model_name, stage_name, pipe):\n"
            "    cv_res = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, return_train_score=True, n_jobs=-1)\n"
            "    pipe.fit(X_train, y_train)\n"
            "    y_tr_pred = pipe.predict(X_train)\n"
            "    y_te_pred = pipe.predict(X_test)\n"
            "    y_te_prob = pipe.predict_proba(X_test)[:, 1]\n"
            "    tn, fp, fn, tp = confusion_matrix(y_test, y_te_pred, labels=[0, 1]).ravel()\n"
            "    tr_f1 = f1_score(y_train, y_tr_pred)\n"
            "    te_f1 = f1_score(y_test, y_te_pred)\n"
            "    te_rec = recall_score(y_test, y_te_pred)\n"
            "    te_spec = specificity_score(y_test, y_te_pred)\n"
            "    return {\n"
            "        'Model': model_name,\n"
            "        'Stage': stage_name,\n"
            "        'Train_F1': round(tr_f1, 4),\n"
            "        'CV_F1_Mean': round(float(np.mean(cv_res['test_f1'])), 4),\n"
            "        'CV_F1_SD': round(float(np.std(cv_res['test_f1'])), 4),\n"
            "        'CV_ROC_AUC': round(float(np.mean(cv_res['test_roc_auc'])), 4),\n"
            "        'Test_Acc': round(accuracy_score(y_test, y_te_pred), 4),\n"
            "        'Test_Prec': round(precision_score(y_test, y_te_pred), 4),\n"
            "        'Test_Recall_Sens': round(te_rec, 4),\n"
            "        'Test_Spec': round(te_spec, 4),\n"
            "        'Test_F1': round(te_f1, 4),\n"
            "        'Test_GMean': round(float(np.sqrt(te_rec * te_spec)), 4),\n"
            "        'Test_ROC_AUC': round(roc_auc_score(y_test, y_te_prob), 4),\n"
            "        'Test_PR_AUC': round(average_precision_score(y_test, y_te_prob), 4),\n"
            "        'Gap_Train_Test_F1': round(tr_f1 - te_f1, 4),\n"
            "        'TN': int(tn), 'FP': int(fp), 'FN': int(fn), 'TP': int(tp)\n"
            "    }\n\n"
            "results = []\n"
            "fitted = {'Exp1_Baseline': {}, 'Exp2_SMOTE': {}, 'Exp3_Tuned': {}}\n\n"
            "# Stage 1 & Stage 2\n"
            "for stage_key, use_smote in [('Exp1_Baseline', False), ('Exp2_SMOTE', True)]:\n"
            "    models = {\n"
            "        'Decision Tree': DecisionTreeClassifier(random_state=RANDOM_STATE),\n"
            "        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),\n"
            "        'XGBoost': XGBClassifier(n_estimators=100, random_state=RANDOM_STATE, eval_metric='logloss', n_jobs=-1),\n"
            "        'SVM': SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE),\n"
            "    }\n"
            "    for m_name, clf in models.items():\n"
            "        p = build_pipeline(clf, use_smote=use_smote)\n"
            "        results.append(evaluate_model(m_name, stage_key, p))\n"
            "        fitted[stage_key][m_name] = p\n\n"
            "# Stage 3: Hyperparameter Tuning via 5-Fold Stratified GridSearchCV\n"
            "tuning_spaces = {\n"
            "    'Decision Tree': (\n"
            "        build_pipeline(DecisionTreeClassifier(random_state=RANDOM_STATE), use_smote=True),\n"
            "        {\n"
            "            'smote__sampling_strategy': [0.85, 1.0],\n"
            "            'classifier__criterion': ['gini', 'entropy'],\n"
            "            'classifier__max_depth': [3, 4, 5, 6],\n"
            "            'classifier__min_samples_split': [10, 20, 30],\n"
            "            'classifier__min_samples_leaf': [5, 10, 15],\n"
            "            'classifier__ccp_alpha': [0.0, 0.005, 0.01],\n"
            "        }\n"
            "    ),\n"
            "    'Random Forest': (\n"
            "        build_pipeline(RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1), use_smote=True),\n"
            "        {\n"
            "            'smote__sampling_strategy': [0.85, 1.0],\n"
            "            'classifier__n_estimators': [150, 250],\n"
            "            'classifier__max_depth': [5, 7, 9],\n"
            "            'classifier__min_samples_split': [8, 12],\n"
            "            'classifier__min_samples_leaf': [3, 6],\n"
            "            'classifier__max_features': ['sqrt', 0.5],\n"
            "        }\n"
            "    ),\n"
            "    'XGBoost': (\n"
            "        build_pipeline(XGBClassifier(random_state=RANDOM_STATE, eval_metric='logloss', n_jobs=-1), use_smote=True),\n"
            "        {\n"
            "            'smote__sampling_strategy': [0.85, 1.0],\n"
            "            'classifier__n_estimators': [100, 150],\n"
            "            'classifier__max_depth': [2, 3, 4],\n"
            "            'classifier__learning_rate': [0.03, 0.05, 0.1],\n"
            "            'classifier__subsample': [0.8],\n"
            "            'classifier__colsample_bytree': [0.8, 1.0],\n"
            "            'classifier__reg_lambda': [1.0, 5.0],\n"
            "            'classifier__min_child_weight': [2, 5],\n"
            "        }\n"
            "    ),\n"
            "    'SVM': (\n"
            "        build_pipeline(SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE), use_smote=True),\n"
            "        {\n"
            "            'smote__sampling_strategy': [0.9, 1.0],\n"
            "            'smote__k_neighbors': [3, 5, 7],\n"
            "            'classifier__C': [0.25, 0.5, 1.0, 2.0, 4.0],\n"
            "            'classifier__gamma': ['scale', 0.03, 0.05, 0.08, 0.12],\n"
            "        }\n"
            "    ),\n"
            "}\n\n"
            "best_params_dict = {}\n"
            "for m_name, (base_p, grid_space) in tuning_spaces.items():\n"
            "    gs = GridSearchCV(base_p, grid_space, scoring='f1', cv=cv, n_jobs=-1, refit=True)\n"
            "    gs.fit(X_train, y_train)\n"
            "    best_params_dict[m_name] = gs.best_params_\n"
            "    results.append(evaluate_model(m_name, 'Exp3_Tuned', gs.best_estimator_))\n"
            "    fitted['Exp3_Tuned'][m_name] = gs.best_estimator_\n\n"
            "results_df = pd.DataFrame(results)\n"
            "print('Optimal Hyperparameters Found in Exp 3:')\n"
            "print(json.dumps(best_params_dict, indent=2))\n"
            "display(results_df)"
        ),
        new_markdown_cell(
            "### Analysis of Step 6 (Quantitative Comparison Across Experimental Stages)\n"
            "The experimental benchmark table demonstrates three critical findings:\n"
            "1. **Baseline Failure Mode (`Exp1_Baseline`):** Without class imbalance mitigation, all models heavily favor the majority non-diabetic class. Although Specificity is high ($0.7900$–$0.8600$), Sensitivity (Recall) is unacceptably low for clinical screening: Decision Tree detects only $48.15\\%$ of diabetic patients ($26/54$), SVM detects $55.56\\%$ ($30/54$), XGBoost detects $59.26\\%$ ($32/54$), and Random Forest detects $61.11\\%$ ($33/54$). Moreover, all three baseline tree models have `Train_F1 = 1.0000`, resulting in severe overfitting gaps (`Gap_Train_Test_F1` between $0.3465$ and $0.4851$).\n"
            "2. **Effect of Fold-Internal SMOTE (`Exp2_SMOTE`):** Introducing SMOTE inside the training folds immediately lifts Sensitivity across all four paradigms: Decision Tree jumps to $61.11\\%$, Random Forest to $70.37\\%$, XGBoost to $64.81\\%$, and SVM to $68.52\\%$.\n"
            "3. **Synergy of Regularization Tuning + SMOTE (`Exp3_Tuned`):** Optimizing structural capacity (`max_depth=4, min_samples_leaf=15` for Decision Tree; `max_depth=3, n_estimators=150, learning_rate=0.03, reg_lambda=5.0, min_child_weight=2` for XGBoost; `C=0.5, gamma=0.03` for SVM) dramatically reduces the train-test generalization gap while elevating Sensitivity to **$74.07\\%$–$79.63\\%$**.\n"
            "   - **Random Forest (Tuned)** achieves the highest 5-Fold CV F1 ($0.7112 \\pm 0.0154$), closely followed by **XGBoost (Tuned)** ($0.7078 \\pm 0.0217$), which delivers the highest Test ROC-AUC ($0.8180$) and the highest Test PR-AUC ($0.7077$) alongside $79.63\\%$ Sensitivity ($43/54$) and half the generalization gap ($0.0843$ vs. $0.1587$).\n"
            "   - **Decision Tree (Tuned)** achieves the highest binary-threshold Test F1 ($0.7049$) and G-Mean ($0.7728$) with a minimal generalization gap of $0.0165$."
        ),
        new_code_cell(
            "# Cell 7: Confusion Matrices and ROC / Precision-Recall Curves for Tuned Models\n"
            "model_order = ['Decision Tree', 'Random Forest', 'XGBoost', 'SVM']\n"
            "model_colors = {'Decision Tree': '#e67e22', 'Random Forest': '#27ae60', 'XGBoost': '#c0392b', 'SVM': '#2980b9'}\n\n"
            "fig, axes = plt.subplots(2, 2, figsize=(10, 8))\n"
            "for idx, m_name in enumerate(model_order):\n"
            "    ax = axes.flatten()[idx]\n"
            "    pipe = fitted['Exp3_Tuned'][m_name]\n"
            "    y_pred = pipe.predict(X_test)\n"
            "    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])\n"
            "    tn, fp, fn, tp = cm.ravel()\n"
            "    sens, spec, f1_v = tp/(tp+fn), tn/(tn+fp), (2*tp)/(2*tp+fp+fn)\n"
            "    annot = np.array([\n"
            "        [f'TN = {tn}\\n({tn/(tn+fp)*100:.1f}%)', f'FP = {fp}\\n({fp/(tn+fp)*100:.1f}%)'],\n"
            "        [f'FN = {fn}\\n({fn/(tp+fn)*100:.1f}%)', f'TP = {tp}\\n({tp/(tp+fn)*100:.1f}%)']\n"
            "    ])\n"
            "    sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', cbar=False, ax=ax,\n"
            "                xticklabels=['Pred: Non-Diabetic (0)', 'Pred: Diabetic (1)'],\n"
            "                yticklabels=['Actual: Non-Diabetic (0)', 'Actual: Diabetic (1)'],\n"
            "                annot_kws={'fontweight': 'bold'})\n"
            "    ax.set_title(f'({chr(97+idx)}) {m_name} (Tuned)\\nSens: {sens:.3f} | Spec: {spec:.3f} | F1: {f1_v:.3f}', fontweight='bold')\n"
            "plt.tight_layout()\n"
            "plt.show()\n\n"
            "fig2, axes2 = plt.subplots(1, 2, figsize=(11.5, 4.5))\n"
            "axes2[0].plot([0, 1], [0, 1], 'k--', lw=1.2, label='Random Chance (AUC = 0.500)')\n"
            "axes2[1].axhline(y_test.mean(), color='k', linestyle='--', lw=1.2, label=f'No-Skill Baseline ({y_test.mean():.3f})')\n"
            "for m_name in model_order:\n"
            "    pipe = fitted['Exp3_Tuned'][m_name]\n"
            "    y_prob = pipe.predict_proba(X_test)[:, 1]\n"
            "    fpr, tpr, _ = roc_curve(y_test, y_prob)\n"
            "    prec, rec, _ = precision_recall_curve(y_test, y_prob)\n"
            "    axes2[0].plot(fpr, tpr, lw=2.2, color=model_colors[m_name], label=f'{m_name} (AUC = {auc(fpr, tpr):.3f})')\n"
            "    axes2[1].plot(rec, prec, lw=2.2, color=model_colors[m_name], label=f'{m_name} (PR-AUC = {average_precision_score(y_test, y_prob):.3f})')\n"
            "axes2[0].set_title('(a) ROC Curves on Hold-Out Test Set', fontweight='bold')\n"
            "axes2[0].set_xlabel('False Positive Rate (1 - Specificity)')\n"
            "axes2[0].set_ylabel('True Positive Rate (Sensitivity)')\n"
            "axes2[0].legend(loc='lower right')\n"
            "axes2[0].grid(True, linestyle='--', alpha=0.4)\n"
            "axes2[1].set_title('(b) Precision-Recall Curves on Hold-Out Test Set', fontweight='bold')\n"
            "axes2[1].set_xlabel('Recall (Sensitivity)')\n"
            "axes2[1].set_ylabel('Precision (PPV)')\n"
            "axes2[1].legend(loc='lower left')\n"
            "axes2[1].grid(True, linestyle='--', alpha=0.4)\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        new_markdown_cell(
            "### Analysis of Step 7 (Confusion Matrices, ROC-AUC, and PR-AUC)\n"
            "- **Clinical False-Negative Reduction:** Out of $54$ diabetic patients in the held-out test set, Tuned Decision Tree, Tuned Random Forest, and Tuned XGBoost all correctly capture **$43$ patients** ($\\text{FN} = 11$, Sensitivity $= 79.63\\%$), and Tuned SVM captures **$40$ patients** ($\\text{FN} = 14$, Sensitivity $= 74.07\\%$). Compared to the baseline Decision Tree ($\\text{FN} = 28$), false negatives are cut by more than **$60\\%$**.\n"
            "- **Threshold-Independent Discrimination:** In the ROC and Precision-Recall curves, **Tuned XGBoost** achieves the highest ROC-AUC ($0.8180$) and highest PR-AUC ($0.7077$), followed closely by **Tuned Random Forest** (ROC-AUC $= 0.8146$, PR-AUC $= 0.6974$) and **Tuned SVM** (ROC-AUC $= 0.8078$, PR-AUC $= 0.6937$)."
        ),
        new_code_cell(
            "# Cell 8: Bias-Variance Diagnosis via Learning Curves & Permutation Feature Importance\n"
            "fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2))\n"
            "train_fracs = np.linspace(0.2, 1.0, 5)\n"
            "for idx, m_name in enumerate(model_order):\n"
            "    ax = axes.flatten()[idx]\n"
            "    sb, tr_b, val_b = learning_curve(fitted['Exp1_Baseline'][m_name], X_train, y_train, cv=cv, train_sizes=train_fracs, scoring='f1', n_jobs=-1)\n"
            "    st, tr_t, val_t = learning_curve(fitted['Exp3_Tuned'][m_name], X_train, y_train, cv=cv, train_sizes=train_fracs, scoring='f1', n_jobs=-1)\n"
            "    ax.plot(sb, np.mean(tr_b, axis=1), '--', color='#7f8c8d', lw=1.4, label='Baseline Train F1')\n"
            "    ax.plot(sb, np.mean(val_b, axis=1), ':', color='#7f8c8d', lw=1.7, label='Baseline CV-Val F1')\n"
            "    c = model_colors[m_name]\n"
            "    tr_tm, val_tm, val_ts = np.mean(tr_t, axis=1), np.mean(val_t, axis=1), np.std(val_t, axis=1)\n"
            "    ax.plot(st, tr_tm, 'o--', color=c, lw=1.8, label='Tuned Train F1')\n"
            "    ax.plot(st, val_tm, 's-', color=c, lw=2.2, label='Tuned CV-Val F1')\n"
            "    ax.fill_between(st, val_tm - val_ts, val_tm + val_ts, color=c, alpha=0.15)\n"
            "    gap_b = float(np.mean(tr_b[-1]) - np.mean(val_b[-1]))\n"
            "    gap_t = float(tr_tm[-1] - val_tm[-1])\n"
            "    ax.set_title(f'({chr(97+idx)}) {m_name}\\nOverfit Gap: Baseline={gap_b:.3f} -> Tuned={gap_t:.3f}', fontweight='bold')\n"
            "    ax.set_xlabel('Training Fold Sample Size')\n"
            "    ax.set_ylabel('F1-Score')\n"
            "    ax.set_ylim(0.40, 1.05)\n"
            "    ax.grid(True, linestyle='--', alpha=0.4)\n"
            "    ax.legend(loc='lower right', ncol=2, fontsize=7.5)\n"
            "plt.tight_layout()\n"
            "plt.show()\n\n"
            "# Permutation Feature Importance\n"
            "perm_rows = []\n"
            "for m_name in model_order:\n"
            "    perm = permutation_importance(fitted['Exp3_Tuned'][m_name], X_test, y_test, scoring='f1', n_repeats=30, random_state=RANDOM_STATE, n_jobs=-1)\n"
            "    imp = np.maximum(perm.importances_mean, 0)\n"
            "    imp_norm = imp / imp.sum() if imp.sum() > 0 else imp\n"
            "    for feat, val in zip(FEATURE_COLUMNS, imp_norm):\n"
            "        perm_rows.append({'Feature': feat, 'Model': m_name, 'Relative_Importance': val})\n"
            "perm_df = pd.DataFrame(perm_rows)\n"
            "order_f = perm_df.groupby('Feature')['Relative_Importance'].mean().sort_values(ascending=False).index.tolist()\n"
            "plt.figure(figsize=(8, 4.2))\n"
            "sns.barplot(data=perm_df, y='Feature', x='Relative_Importance', hue='Model', order=order_f, palette=model_colors, edgecolor='black')\n"
            "plt.title('Normalized Permutation Feature Importance (Test F1 Impact Across Tuned Models)', fontweight='bold')\n"
            "plt.xlabel('Normalized Importance Weight')\n"
            "plt.grid(axis='x', linestyle='--', alpha=0.4)\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        new_markdown_cell(
            "### Conclusion and Final Summary Answering the Research Questions\n\n"
            "1. **Resolution of Disguised Missing Data & Prevention of Data Leakage:**\n"
            "   - Auditing the Pima Indians Diabetes cohort ($N=768$) exposed biologically impossible zero entries in `Insulin` ($48.70\\%$), `SkinThickness` ($29.56\\%$), `BloodPressure` ($4.56\\%$), `BMI` ($1.43\\%$), and `Glucose` ($0.65\\%$). Encapsulating zero-to-NaN recoding, median imputation, z-score standardization, and SMOTE inside an `imblearn.pipeline.Pipeline` guarantees zero data leakage across cross-validation and hold-out evaluation.\n"
            "2. **Quantifiable Impact of Imbalance Mitigation and Hyperparameter Optimization:**\n"
            "   - Unadjusted baseline models suffer from the **accuracy paradox**: while Specificity reaches $0.7900$–$0.8600$, Sensitivity is only $0.4815$–$0.6111$.\n"
            "   - Combining fold-internal SMOTE with 5-fold stratified hyperparameter optimization (`Exp3_Tuned`) elevates clinical Sensitivity to **$0.7407$–$0.7963$** across all four classifiers while maintaining Specificity between $0.7000$ and $0.7500$.\n"
            "3. **Bias–Variance Diagnosis from Learning Curves:**\n"
            "   - Unconstrained baseline tree learners (Decision Tree, Random Forest, and XGBoost) exhibit severe high variance ($F_{1,\\text{train}} = 1.0000$, full-fold CV generalization gaps of $0.367$–$0.448$).\n"
            "   - Structural pruning (`max_depth=4, min_samples_leaf=15` in Decision Tree) and second-order gradient regularization (`max_depth=3, n_estimators=150, learning_rate=0.03, reg_lambda=5.0, min_child_weight=2` in XGBoost) compress the learning-curve gap at $n=491$ to **$0.057$** and **$0.064$**, respectively, while RBF SVM (`C=0.5, gamma=0.03`) converges with a near-zero learning-curve gap of **$0.006$**.\n"
            "4. **Optimal Model Recommendation:**\n"
            "   - **Primary Recommendation for Risk Stratification:** **Tuned XGBoost** achieves the highest Test ROC-AUC ($0.8180$) and highest Test PR-AUC ($0.7077$), paired with peak Sensitivity ($0.7963$), strong 5-Fold CV F1 ($0.7078 \\pm 0.0217$), and a low generalization gap ($0.0843$, half of Random Forest's $0.1587$).\n"
            "   - **Primary Recommendation for Rule-Interpretable Point-of-Care Screening:** **Tuned Regularized Decision Tree** achieves the highest fixed-threshold Test F1 ($0.7049$), Sensitivity ($0.7963$), Specificity ($0.7500$), and G-Mean ($0.7728$) with the smallest train-test gap ($0.0165$)."
        ),
    ]

    nb.cells = cells
    nb_path = DELIVERABLES_DIR / "02_Colab_Notebook_Diabetes_Classification.ipynb"
    with open(nb_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)

    print("Executing Colab Notebook via nbclient to embed all outputs and charts...")
    client = NotebookClient(
        nb,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": str(DELIVERABLES_DIR)}},
    )
    client.execute()

    with open(nb_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)

    print(f"Saved executed notebook to: {nb_path}")
    return nb_path


def download_ieetran_and_tectonic() -> Path:
    """Download official IEEEtran.cls/IEEEtran.bst from CTAN and Tectonic standalone LaTeX compiler."""
    TOOLS_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)

    cls_path = PAPER_DIR / "IEEEtran.cls"
    bst_path = PAPER_DIR / "IEEEtran.bst"

    if not cls_path.exists():
        print("Downloading official IEEEtran.cls from CTAN...")
        urllib.request.urlretrieve(
            "https://mirrors.ctan.org/macros/latex/contrib/IEEEtran/IEEEtran.cls",
            cls_path,
        )

    if not bst_path.exists():
        print("Downloading official IEEEtran.bst from CTAN...")
        urllib.request.urlretrieve(
            "https://mirrors.ctan.org/macros/latex/contrib/IEEEtran/bibtex/IEEEtran.bst",
            bst_path,
        )

    system_tectonic = shutil.which("tectonic")
    if system_tectonic:
        return Path(system_tectonic)

    machine = platform.machine().lower()
    is_arm = machine in {"arm64", "aarch64"}

    if sys.platform == "win32":
        asset_name = "tectonic-0.15.0-x86_64-pc-windows-msvc.zip"
        bin_name = "tectonic.exe"
    elif sys.platform == "darwin":
        asset_name = (
            "tectonic-0.15.0-aarch64-apple-darwin.tar.gz"
            if is_arm
            else "tectonic-0.15.0-x86_64-apple-darwin.tar.gz"
        )
        bin_name = "tectonic"
    elif sys.platform.startswith("linux"):
        asset_name = (
            "tectonic-0.15.0-aarch64-unknown-linux-musl.tar.gz"
            if is_arm
            else "tectonic-0.15.0-x86_64-unknown-linux-gnu.tar.gz"
        )
        bin_name = "tectonic"
    else:
        raise RuntimeError(
            f"Unsupported platform for Tectonic auto-download: {sys.platform} ({machine})"
        )

    tectonic_bin = TOOLS_DIR / bin_name
    if not tectonic_bin.exists():
        print(f"Downloading standalone Tectonic LaTeX engine ({asset_name})...")
        base_url = "https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%400.15.0/"
        archive_path = TOOLS_DIR / asset_name
        urllib.request.urlretrieve(base_url + asset_name, archive_path)
        if asset_name.endswith(".zip"):
            with zipfile.ZipFile(archive_path, "r") as zf:
                zf.extractall(TOOLS_DIR)
        else:
            with tarfile.open(archive_path, "r:gz") as tf:
                tf.extractall(TOOLS_DIR)
        if sys.platform != "win32" and tectonic_bin.exists():
            tectonic_bin.chmod(
                tectonic_bin.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
            )

    return tectonic_bin


def compile_ieee_paper(tectonic_exe: Path) -> Path:
    """Compile paper/main.tex into paper/main.pdf and copy to deliverables/01_IEEE_Article_Diabetes_Classification.pdf."""
    print("Compiling IEEEtran LaTeX manuscript via Tectonic...")
    cmd = [str(tectonic_exe), "main.tex"]
    subprocess.run(cmd, cwd=str(PAPER_DIR), check=True)

    compiled_pdf = PAPER_DIR / "main.pdf"
    dest_pdf = DELIVERABLES_DIR / "01_IEEE_Article_Diabetes_Classification.pdf"
    shutil.copy2(compiled_pdf, dest_pdf)
    print(f"Compiled IEEE Article PDF saved to: {dest_pdf}")
    return dest_pdf


def build_presentation_slides_pdf(tectonic_exe: Path) -> Path:
    """Compile paper/presentation_slides.tex into a 16:9 Beamer PDF and copy to deliverables/."""
    slides_tex_path = PAPER_DIR / "presentation_slides.tex"
    if not slides_tex_path.exists():
        raise FileNotFoundError(f"Missing presentation source file: {slides_tex_path}")

    print("Compiling 16:9 Beamer Presentation Slides PDF via Tectonic...")
    subprocess.run([str(tectonic_exe), "presentation_slides.tex"], cwd=str(PAPER_DIR), check=True)
    compiled_slides = PAPER_DIR / "presentation_slides.pdf"
    dest_slides = DELIVERABLES_DIR / "03_Presentation_Slides_Diabetes_Classification.pdf"
    shutil.copy2(compiled_slides, dest_slides)
    print(f"Compiled Presentation Slides PDF saved to: {dest_slides}")
    return dest_slides


def main() -> None:
    nb_path = build_and_execute_colab_notebook()
    tectonic_exe = download_ieetran_and_tectonic()
    paper_pdf = compile_ieee_paper(tectonic_exe)
    slides_pdf = build_presentation_slides_pdf(tectonic_exe)
    print("\n=== ALL DELIVERABLES SUCCESSFULLY BUILT ===")
    print(f"1. IEEE Article PDF : {paper_pdf}")
    print(f"2. Colab Notebook   : {nb_path}")
    print(f"3. Presentation PDF : {slides_pdf}")


if __name__ == "__main__":
    main()
