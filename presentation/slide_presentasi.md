# Slide Presentasi Video (Durasi Maksimal 5 Menit)
**Judul Penelitian:** Comparative Evaluation of Tree-Based, Ensemble, and Margin Classifiers for Early Diabetes Mellitus Prediction Under Zero-Leakage Preprocessing and Class Imbalance  
**Peneliti:** Mokhammad Dwi Nurkolis (NIM: 26917015)  
**Program Studi:** Magister Informatika, Fakultas Teknologi Industri, Universitas Islam Indonesia  

---

## Slide 1 — Judul & Konteks Penelitian (00:00 – 00:45)
* **Permasalahan Klinis:** Deteksi dini *Type 2 Diabetes Mellitus* (T2DM) sangat krusial untuk mencegah komplikasi kronis (retinopati, nefropati, penyakit kardiovaskular).
* **Tiga Masalah Utama pada Studi ML Sebelumnya:**
  1. **Disguised Missing Values:** Nilai `0` yang mustahil secara biologis pada *Insulin* (48,7%), *Skin Thickness* (29,6%), *Blood Pressure* (4,6%), *BMI* (1,4%), dan *Glucose* (0,7%) sering diperlakukan sebagai angka valid.
  2. **Class Imbalance (1 : 1,87):** Model *baseline* terjebak pada *Accuracy Paradox* — *Specificity* tinggi tetapi *Sensitivity/Recall* rendah (banyak pasien diabetes terlewat / *False Negative* tinggi).
  3. **Data Leakage:** Imputasi dan SMOTE yang dilakukan sebelum *train-test split* / di luar *cross-validation fold* membuat skor terlihat tinggi secara semu namun gagal saat diuji pada pasien baru.

---

## Slide 2 — Karakteristik Dataset & Audit Kualitas Data (00:45 – 01:35)
* **Sumber Dataset:** *Pima Indians Diabetes Database* (OpenML ID 37 / NIDDK, $N = 768$ pasien, 8 fitur klinis numerik, target biner `Outcome`).
* **Distribusi Target:**
  * Non-Diabetic (`0`): **500 pasien (65,10%)**
  * Diabetic (`1`): **268 pasien (34,90%)**
* **Audit Nilai `0` Biologis Mustahil:**
  * `Insulin`: **374 sampel (48,70%)** bernilai 0 $\rightarrow$ rata-rata mentah terdistorsi ($79,80$ vs median valid $125,00\text{ }\mu\text{U/mL}$).
  * `SkinThickness`: **227 sampel (29,56%)** bernilai 0.
  * `BloodPressure`: **35 sampel (4,56%)**, `BMI`: **11 sampel (1,43%)**, `Glucose`: **5 sampel (0,65%)**.
* *(Tampilkan Gambar `fig1_class_and_missing_audit.png` dan `fig2_correlation_heatmap.png`)*

---

## Slide 3 — Metodologi & Arsitektur Pipeline Anti-Data-Leakage (01:35 – 02:35)
* **Pembagian Data:** *Stratified Train-Test Split* 80:20 di awal:
  * **Train Set:** $N = 614$ (400 negatif, 214 positif) $\rightarrow$ digunakan untuk *5-Fold Stratified Cross-Validation*.
  * **Hold-Out Test Set:** $N = 154$ (100 negatif, 54 positif) $\rightarrow$ sama sekali tidak disentuh selama *preprocessing* maupun *tuning*.
* **Pipeline `imblearn` di Dalam Tiap Training Fold:**
  1. `BiologicalZeroToNanTransformer`: Mengubah angka `0` pada 5 fitur klinis menjadi `NaN`.
  2. `SimpleImputer(strategy='median')`: Imputasi median dihitung **hanya dari training fold**.
  3. `StandardScaler()`: Standarisasi $z$-score dihitung **hanya dari training fold**.
  4. `SMOTE`: Sintesis sampel minoritas dilakukan **hanya di dalam training fold** (fold validasi & test tetap murni data asli).
* **4 Algoritma & 3 Tahap Eksperimen:**
  * **Algoritma:** Decision Tree (CART), Random Forest (Bagging), XGBoost (Gradient Boosting), dan SVM (RBF Kernel).
  * **Exp 1 (Baseline):** Parameter default, tanpa SMOTE.
  * **Exp 2 (Imbalance Handling):** Penambahan fold-internal SMOTE.
  * **Exp 3 (Hyperparameter Tuning + SMOTE):** Optimasi *5-Fold Stratified GridSearchCV* berbasis F1-Score.

---

## Slide 4 — Hasil Eksperimen & Analisis Perbandingan Model (02:35 – 03:45)
| Tahap Eksperimen | Algoritma | CV F1 ($\mu \pm \sigma$) | Test Acc | Test Prec | Test Recall (Sens) | Test Spec | Test F1 | Test ROC-AUC | Train-Test Gap ($\Delta F_1$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Exp 1: Baseline** | Decision Tree | $0.5518 \pm 0.0201$ | 0.6818 | 0.5532 | **0.4815** | 0.7900 | 0.5149 | 0.6357 | +0.4851 |
| | Random Forest | $0.6330 \pm 0.0192$ | 0.7727 | 0.7021 | 0.6111 | 0.8600 | 0.6535 | 0.8181 | +0.3465 |
| | XGBoost | $0.6117 \pm 0.0183$ | 0.7662 | 0.6957 | 0.5926 | 0.8600 | 0.6400 | 0.8057 | +0.3600 |
| | SVM (RBF) | $0.6485 \pm 0.0187$ | 0.7403 | 0.6522 | 0.5556 | 0.8400 | 0.6000 | 0.7964 | +0.1461 |
| **Exp 2: SMOTE** | Decision Tree | $0.6144 \pm 0.0100$ | 0.7013 | 0.5690 | 0.6111 | 0.7500 | 0.5893 | 0.6806 | +0.4107 |
| | Random Forest | $0.6683 \pm 0.0356$ | 0.7532 | 0.6333 | 0.7037 | 0.7800 | 0.6667 | 0.8139 | +0.3333 |
| | XGBoost | $0.6063 \pm 0.0277$ | 0.7338 | 0.6140 | 0.6481 | 0.7800 | 0.6306 | 0.8106 | +0.3694 |
| | SVM (RBF) | $0.6854 \pm 0.0432$ | 0.7273 | 0.5968 | 0.6852 | 0.7500 | 0.6379 | 0.8112 | +0.1551 |
| **Exp 3: Tuned+SMOTE** | **Decision Tree** | $0.6739 \pm 0.0246$ | 0.7662 | 0.6324 | **0.7963** | **0.7500** | **0.7049** | 0.8012 | **+0.0165** |
| | **Random Forest** | $0.7112 \pm 0.0154$ | 0.7338 | 0.5890 | **0.7963** | 0.7000 | 0.6772 | 0.8146 | +0.1587 |
| | **XGBoost** | **0.7121 $\pm$ 0.0209** | 0.7338 | 0.5915 | 0.7778 | 0.7100 | 0.6720 | **0.8224** | +0.0811 |
| | **SVM (RBF)** | $0.6956 \pm 0.0091$ | 0.7273 | 0.5882 | 0.7407 | 0.7200 | 0.6557 | 0.8078 | +0.0486 |

* *(Tampilkan Gambar `fig3_confusion_matrices.png` dan `fig4_roc_pr_curves.png`)*

---

## Slide 5 — Analisis Bias–Variance (Learning Curves) & Kesimpulan (03:45 – 05:00)
* **Diagnosis Bias–Variance (`fig5_learning_curves_bias_variance.png`):**
  * Pada **Exp 1 (Baseline)**, Decision Tree, Random Forest, dan XGBoost mengalami *severe overfitting* (*High Variance*): skor Train F1 = `1.0000` sedangkan Test F1 hanya `0.5149–0.6535` (Gap $\Delta F_1$ mencapai `0.34–0.48`).
  * Pada **Exp 3 (Tuned)**, pembatasan kapasitas (`max_depth=4, min_samples_leaf=15` pada Decision Tree; `max_depth=4, learning_rate=0.03, reg_lambda=5.0` pada XGBoost) berhasil memangkas *generalization gap* menjadi **`0.0165`** (DT) dan **`0.0811`** (XGBoost).
* **Faktor Klinis Dominan (`fig6_feature_importance.png`):**
  * Seluruh model sepakat bahwa **Plasma Glucose** (47%–58% bobot), **BMI**, dan **Age** adalah penentu utama diagnosis diabetes.
* **Rekomendasi Pemilihan Model:**
  * **XGBoost (Tuned)** dipilih sebagai model terbaik untuk **probabilistic clinical risk stratification** karena unggul pada **5-Fold CV F1 (`0.7121`)**, **Test ROC-AUC (`0.8224`)**, dan **Test PR-AUC (`0.7125`)**.
  * **Decision Tree (Tuned, `max_depth=4`)** sangat ideal untuk **skrining cepat berbasis aturan yang mudah diinterpretasi dokter** karena mencapai **Sensitivity `79,63%`** (hanya 11 *False Negative* dari 54 pasien) dan **Test F1 `0.7049`** dengan *overfitting gap* terkecil (`0.0165`).
