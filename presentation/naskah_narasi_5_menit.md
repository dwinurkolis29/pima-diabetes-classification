# Naskah Narasi Video Presentasi (Durasi: 4 Menit 45 Detik)

> **Tips Rekaman:** Anda dapat menggunakan OBS Studio, Zoom (Record Local), atau Loom. Tampilkan slide PDF (`03_Presentation_Slides_Diabetes_Classification.pdf`) dan Notebook Google Colab secara bergantian sesuai penanda waktu di bawah ini.

---

### [00:00 – 00:45] Pembukaan & Latar Belakang Masalah (Tampilkan Slide 1)
"Assalamu'alaikum warahmatullahi wabarakatuh. Perkenalkan saya Mokhammad Dwi Nurkolis, NIM 26917015, mahasiswa Magister Informatika Universitas Islam Indonesia. Pada presentasi ini saya akan memaparkan hasil penelitian eksperimental kami yang berjudul *Comparative Evaluation of Tree-Based, Ensemble, and Margin Classifiers for Early Diabetes Mellitus Prediction Under Zero-Leakage Preprocessing and Class Imbalance*.

Permasalahan utama dalam prediksi dini Diabetes Mellitus Tipe 2 pada data klinis tabular seperti *Pima Indians Diabetes Database* ada tiga:
Pertama, adanya *disguised missing values*, yaitu nilai nol yang mustahil secara biologis pada fitur klinis seperti insulin, ketebalan kulit, tekanan darah, BMI, dan glukosa.
Kedua, adanya ketimpangan kelas (*class imbalance*) 65,1% sehat berbanding 34,9% positif diabetes, yang menyebabkan model *baseline* memiliki sensitivitas rendah atau banyak melewatkan pasien yang sebenarnya sakit (*False Negative* tinggi).
Ketiga, banyak literatur sebelumnya melakukan imputasi dan *oversampling* SMOTE sebelum pembagian data latih dan uji, sehingga terjadi *data leakage*."

---

### [00:45 – 01:35] Karakteristik Dataset & Audit Kualitas Data (Tampilkan Slide 2 — Gambar 1)
"Kita menggunakan dataset publik *Pima Indians Diabetes* dari OpenML ID 37 dengan total 768 pasien dan 8 fitur klinis.
Pada tahap eksplorasi data di Gambar 1, kami menemukan bahwa meskipun tidak ada nilai `NULL` secara eksplisit, terdapat 374 pasien atau **48,7%** yang memiliki nilai *Serum Insulin* sama dengan nol, dan **29,6%** memiliki *Skin Thickness* nol. Jika angka nol ini langsung dilatih tanpa diubah menjadi `NaN`, rata-rata dan korelasi klinis menjadi rusak. Karena distribusi fitur klinis seperti Insulin miring ke kanan dengan *outlier* ekstrem, imputasi median jauh lebih tangguh dibandingkan rata-rata."

---

### [01:35 – 02:35] Metodologi Zero-Leakage & Eksplorasi Korelasi (Tampilkan Slide 3 — Gambar 2 / Notebook Colab)
"Untuk menjamin nol *data leakage*, kami membagi dataset di awal menggunakan *Stratified Train-Test Split* 80 banding 20, menghasilkan 614 sampel latih dan 154 sampel uji.
Pada Gambar 2, analisis korelasi pasca-imputasi menunjukkan bahwa *Plasma Glucose*, *BMI*, dan *Age* memiliki hubungan paling kuat terhadap status diabetes.
Seluruh tahapan pra-pemrosesan—mulai dari konversi nilai nol biologis menjadi `NaN`, imputasi median, standarisasi *StandardScaler*, hingga sintesis data minoritas dengan SMOTE—dibungkus ke dalam satu objek `imblearn.Pipeline`. Dengan arsitektur ini, saat *5-Fold Stratified Cross-Validation* berjalan, parameter median, rata-rata, maupun sampel sintetis SMOTE murni dihitung dari *training fold* saja dan tidak pernah bocor ke data validasi maupun *test set*.

Kami membandingkan empat paradigma algoritma—yaitu Decision Tree, Random Forest, XGBoost, dan SVM kernel RBF—dalam tiga tahap eksperimen terukur:
1. **Eksperimen 1 (Baseline):** Parameter default tanpa penanganan ketimpangan kelas.
2. **Eksperimen 2 (Imbalance Handling):** Penambahan SMOTE di dalam *training fold*.
3. **Eksperimen 3 (Hyperparameter Tuning):** Optimasi *GridSearchCV* 5-Fold Stratified bersama SMOTE untuk mengendalikan *overfitting* dan memaksimalkan F1-Score."

---

### [02:35 – 03:15] Tabel Hasil Eksperimen 3 Tahap (Tampilkan Slide 4)
"Hasil kuantitatif pada *hold-out test set* sebanyak 154 pasien menunjukkan pola yang sangat jelas:
Pada **Eksperimen 1 (Baseline)**, semua model mengalami *accuracy paradox*. Random Forest dan XGBoost mencapai *Specificity* 86% dan akurasi sekitar 77%, tetapi *Sensitivity* atau *Recall*-nya hanya 59% sampai 61%. Bahkan Decision Tree *baseline* hanya mendeteksi 48,15% pasien diabetes—artinya lebih dari separuh pasien sakit gagal terdeteksi.
Pada **Eksperimen 2**, penambahan SMOTE langsung mengangkat *Sensitivity* di keempat model: Decision Tree naik ke 61,11%, Random Forest ke 70,37%, XGBoost ke 64,81%, dan SVM ke 68,52%.
Pada **Eksperimen 3**, kombinasi SMOTE dan *hyperparameter tuning* menghasilkan keseimbangan terbaik antara sensitivitas dan generalisasi."

---

### [03:15 – 03:55] Confusion Matrix & Kurva Diskriminasi ROC/PR (Tampilkan Slide 5 — Gambar 3 & Gambar 4)
"Jika dilihat lebih rinci pada *Confusion Matrix* di Gambar 3, ketiga model berbasis pohon hasil tuning—**Decision Tree** (`max_depth=4`, `min_samples_leaf=15`), **Random Forest**, dan **XGBoost**—berhasil menangkap **43 dari 54 pasien diabetes** (*Sensitivity* **79,63%**), menurunkan *False Negative* dari 28 kasus pada baseline menjadi hanya 11 kasus, dengan Test F1-score tertinggi **0,7049** diraih oleh Decision Tree.
Sementara itu, pada kurva ROC dan *Precision-Recall* di Gambar 4, **XGBoost hasil tuning** meraih performa pemeringkatan risiko tertinggi secara keseluruhan di berbagai ambang batas, dengan **Test ROC-AUC tertinggi sebesar 0,8180**, **Test PR-AUC tertinggi sebesar 0,7077**, serta **5-Fold CV F1 sebesar 0,7078** (mendekati puncak CV F1 Random Forest sebesar **0,7112**)."

---

### [03:55 – 05:00] Analisis Bias–Variance, Feature Importance & Kesimpulan (Tampilkan Slide 6 — Gambar 5 & Gambar 6)
"Mengapa peningkatan pada Eksperimen 3 sangat signifikan? Jawabannya terlihat pada *Learning Curves* Bias–Variance di Gambar 5.
Pada kondisi *baseline*, Decision Tree, Random Forest, dan XGBoost mengalami *severe overfitting* dengan skor Train F1 sempurna `1,0000` tetapi jatuh di data uji (*gap* 0,35 hingga 0,49). Melalui regularisasi struktural—yaitu pembatasan kedalaman pohon `max_depth=4` dan `min_samples_leaf=15` pada Decision Tree, serta `max_depth=3`, `learning_rate=0.03`, dan regularisasi $L_2$ `reg_lambda=5.0` pada XGBoost—*Train–Test gap* berhasil ditekan drastis menjadi **0,0165** pada Decision Tree dan **0,0843** pada XGBoost.
Analisis *Permutation Feature Importance* pada Gambar 6 juga mengonfirmasi konsensus keempat model bahwa *Plasma Glucose*, *BMI*, dan *Age* merupakan prediktor klinis utama.

Sebagai kesimpulan:
1. Untuk kebutuhan **pemeringkatan skor risiko klinis** di berbagai ambang batas keputusan, **Tuned XGBoost** adalah model terbaik karena unggul pada ROC-AUC (`0,8180`) dan PR-AUC (`0,7077`), sekaligus menjaga *Sensitivity* `79,63%` dengan generalisasi yang jauh lebih stabil daripada Random Forest (`gap` 0,0843 vs 0,1587).
2. Untuk kebutuhan **skrining cepat yang menuntut aturan keputusan transparan (white-box)**, **Tuned Pruned Decision Tree** sangat direkomendasikan karena memberikan *Sensitivity* `79,63%`, *Specificity* `75,00%`, dan F1-score `0,7049` dengan tingkat *overfitting* paling rendah.

Seluruh kode sumber, notebook Google Colab yang dapat direproduksi satu klik, serta draft artikel IEEE tersedia lengkap pada repositori dan folder deliverables. Terima kasih, wassalamu'alaikum warahmatullahi wabarakatuh."
