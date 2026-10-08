"""
train_models.py
===============
Skrip training untuk sistem penjernihan limbah batik.

Model yang dilatih:
  1. RandomForestRegressor  (multi-output)  → prediksi % penyisihan polutan
  2. RandomForestClassifier (balanced)      → prediksi label_clean (bersih/tidak)
  3. IsolationForest                        → deteksi anomali proses
  4. RandomForestRegressor  (multi-output)  → manajemen energi hibrida

Prasyarat:
  Jalankan generate_dataset.py terlebih dahulu untuk menghasilkan:
    dataset_water_quality.csv
    dataset_energy_management.csv

Output (disimpan di folder models/):
  rf_regressor_removal.pkl   — prediksi 5 nilai penyisihan polutan
  rf_classifier_clean.pkl    — prediksi label_clean 0/1
  isolation_forest.pkl       — deteksi anomali
  scaler_isolation.pkl       — scaler untuk Isolation Forest
  rf_regressor_energy.pkl    — prediksi alokasi + forecast energi
  scaler_energy.pkl          — scaler untuk Energy Management
"""

import os
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import (IsolationForest, RandomForestClassifier,
                              RandomForestRegressor)
from sklearn.metrics import (classification_report, confusion_matrix,
                             mean_absolute_error, r2_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ============================================================
# KONFIGURASI GLOBAL
# ============================================================
RANDOM_STATE  = 42
TEST_SIZE     = 0.20
CONTAMINATION = 0.05   # asumsi ~5% data adalah anomali
MODELS_DIR    = 'models'

os.makedirs(MODELS_DIR, exist_ok=True)


def save_model(obj, filename):
    """Simpan objek ke folder models/ menggunakan pickle."""
    path = os.path.join(MODELS_DIR, filename)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    size_kb = os.path.getsize(path) / 1024
    print(f"  [SAVED] {filename}  ({size_kb:.1f} KB)")


def print_section(title):
    print()
    print("=" * 62)
    print(f"  {title}")
    print("=" * 62)


# ============================================================
# LOAD DATA
# ============================================================
print_section("LOAD DATASET")

for fname in ('dataset_water_quality.csv', 'dataset_energy_management.csv'):
    if not os.path.exists(fname):
        raise FileNotFoundError(
            f"File '{fname}' tidak ditemukan. "
            "Jalankan generate_dataset.py terlebih dahulu."
        )

df_water  = pd.read_csv('dataset_water_quality.csv')
df_energy = pd.read_csv('dataset_energy_management.csv')

print(f"  Water Quality   : {df_water.shape[0]} baris × {df_water.shape[1]} kolom")
print(f"  Energy Mgmt     : {df_energy.shape[0]} baris × {df_energy.shape[1]} kolom")


# ============================================================
# DEFINISI KOLOM
# ============================================================

# 16 kolom input asli dari dataset air (dipakai oleh Isolation Forest)
INPUT_COLS_WATER = [
    'pH_init', 'turbidity_init_NTU', 'TDS_init_mgL', 'conductivity_uScm',
    'color_init_PtCo', 'COD_init_mgL', 'BOD_init_mgL', 'temp_water_C',
    'TSS_init_mgL', 'voltage_V', 'current_A', 'duration_min',
    'electrode_type', 'electrode_gap_cm', 'power_consumption_W',
    'inflow_rate_Lmin',
]

# Target regresi: persentase penyisihan polutan (5 kolom)
TARGET_REMOVAL = [
    'COD_removal_pct', 'BOD_removal_pct', 'TSS_removal_pct',
    'color_removal_pct', 'turbidity_removal_pct',
]

# Target klasifikasi: label bersih/tidak
TARGET_LABEL = 'label_clean'

# Target energi: alokasi sumber + forecast 1 jam (4 kolom)
TARGET_ENERGY = [
    'solar_fraction_pct', 'wind_fraction_pct',
    'battery_fraction_pct', 'energy_forecast_1h_Wh',
]


# ============================================================
# ██  BAGIAN 1: WATER QUALITY — PREPROCESSING
# ============================================================
print_section("BAGIAN 1 — WATER QUALITY: PREPROCESSING")

# --- Feature Engineering: muatan listrik Q ---
# Q (Coulomb) = Arus × Durasi_menit × 60
# Variabel fisika utama yang menggerakkan efisiensi penyisihan.
# Meski power_consumption_W sudah ada, Q menggabungkan durasi
# sehingga memberi informasi kumulatif yang lebih relevan untuk RF.
df_water['Q_coulomb'] = (
    df_water['current_A'] * df_water['duration_min'] * 60
)
print("  [+] Feature baru: Q_coulomb = current × duration × 60 (Coulomb)")

# 17 fitur untuk kedua model RF (16 asli + Q)
FEATURES_WATER = INPUT_COLS_WATER + ['Q_coulomb']
print(f"  Total fitur RF  : {len(FEATURES_WATER)}")

X_water   = df_water[FEATURES_WATER]
y_removal = df_water[TARGET_REMOVAL]   # multi-output, DataFrame
y_label   = df_water[TARGET_LABEL]     # Series, binary

# --- Split untuk Regressor (random biasa) ---
X_tr_reg, X_te_reg, y_tr_reg, y_te_reg = train_test_split(
    X_water, y_removal,
    test_size=TEST_SIZE, random_state=RANDOM_STATE
)

# --- Split untuk Classifier (stratified agar proporsi kelas terjaga) ---
X_tr_clf, X_te_clf, y_tr_clf, y_te_clf = train_test_split(
    X_water, y_label,
    test_size=TEST_SIZE, random_state=RANDOM_STATE,
    stratify=y_label
)

print(f"\n  Regressor  — Train: {len(X_tr_reg):>4}  |  Test: {len(X_te_reg)}")
print(f"  Classifier — Train: {len(X_tr_clf):>4}  |  Test: {len(X_te_clf)}")
print(f"  Label BERSIH (train): {y_tr_clf.sum()} ({y_tr_clf.mean()*100:.1f}%)")
print(f"  Label BERSIH (test) : {y_te_clf.sum()} ({y_te_clf.mean()*100:.1f}%)")
print("  → Catatan: RF tidak butuh StandardScaler (tree-based, tidak sensitif skala)")


# ============================================================
# ██  MODEL 1: RandomForestRegressor — Penyisihan Polutan
# ============================================================
print_section("MODEL 1 — RF Regressor: Prediksi % Penyisihan Polutan")

rf_reg = RandomForestRegressor(
    n_estimators=200,
    max_depth=None,         # biarkan tumbuh penuh, RF memiliki low bias
    min_samples_leaf=2,     # sedikit regularisasi untuk hindari overfit
    n_jobs=-1,
    random_state=RANDOM_STATE,
)
rf_reg.fit(X_tr_reg, y_tr_reg)

y_pred_reg = rf_reg.predict(X_te_reg)

print("\n  Metrik per target (Test Set):")
print(f"  {'Target':<28}  {'MAE':>7}  {'R²':>8}")
print(f"  {'-'*28}  {'-'*7}  {'-'*8}")
for i, col in enumerate(TARGET_REMOVAL):
    mae = mean_absolute_error(y_te_reg[col], y_pred_reg[:, i])
    r2  = r2_score(y_te_reg[col], y_pred_reg[:, i])
    print(f"  {col:<28}  {mae:>6.2f}%  {r2:>8.4f}")

# Feature importance
importances = pd.Series(rf_reg.feature_importances_, index=FEATURES_WATER)
top5 = importances.nlargest(5)
print("\n  Top-5 Feature Importance:")
for feat, val in top5.items():
    bar = '█' * int(val * 200)
    print(f"    {feat:<25} {val:.4f}  {bar}")

save_model(rf_reg, 'rf_regressor_removal.pkl')


# ============================================================
# ██  MODEL 2: RandomForestClassifier — label_clean
# ============================================================
print_section("MODEL 2 — RF Classifier: label_clean (Bersih/Tidak)")
print("  class_weight='balanced' aktif — kompensasi imbalance 1.2% BERSIH")

rf_clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_leaf=2,
    class_weight='balanced',   # wajib: agar kelas minoritas tidak diabaikan
    n_jobs=-1,
    random_state=RANDOM_STATE,
)
rf_clf.fit(X_tr_clf, y_tr_clf)

y_pred_clf = rf_clf.predict(X_te_clf)
y_prob_clf = rf_clf.predict_proba(X_te_clf)[:, 1]

print("\n  Classification Report:")
print(classification_report(
    y_te_clf, y_pred_clf,
    target_names=['Tidak Bersih (0)', 'Bersih (1)'],
    digits=4,
))

cm = confusion_matrix(y_te_clf, y_pred_clf)
tn, fp, fn, tp = cm.ravel()
print(f"  Confusion Matrix:")
print(f"    {'':20s}  Pred: Tidak Bersih  Pred: Bersih")
print(f"    Actual: Tidak Bersih   TN={tn:<6}           FP={fp}")
print(f"    Actual: Bersih         FN={fn:<6}           TP={tp}")

try:
    roc = roc_auc_score(y_te_clf, y_prob_clf)
    print(f"\n  ROC-AUC : {roc:.4f}")
except Exception as e:
    print(f"\n  ROC-AUC tidak dapat dihitung: {e}")

save_model(rf_clf, 'rf_classifier_clean.pkl')


# ============================================================
# ██  MODEL 3: IsolationForest — Anomaly Detection
# ============================================================
print_section("MODEL 3 — Isolation Forest: Deteksi Anomali Proses")
print(f"  Menggunakan {len(INPUT_COLS_WATER)} kolom input asli (tanpa Q_coulomb, tanpa target)")
print("  StandardScaler WAJIB: rentang nilai sangat berbeda")
print("    COD_init: 300–21000 mg/L  vs  electrode_gap: 1–5 cm")

X_iso = df_water[INPUT_COLS_WATER]

scaler_iso   = StandardScaler()
X_iso_scaled = scaler_iso.fit_transform(X_iso)

iso_forest = IsolationForest(
    n_estimators=200,
    contamination=CONTAMINATION,   # asumsi ~5% data adalah anomali
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
iso_forest.fit(X_iso_scaled)

# Evaluasi distribusi skor
scores      = iso_forest.decision_function(X_iso_scaled)
labels_iso  = iso_forest.predict(X_iso_scaled)   # -1=anomali, 1=normal
n_anomaly   = (labels_iso == -1).sum()
n_normal    = (labels_iso ==  1).sum()

print(f"\n  Contamination      : {CONTAMINATION} ({CONTAMINATION*100:.0f}%)")
print(f"  Normal terdeteksi  : {n_normal}  ({n_normal/len(labels_iso)*100:.1f}%)")
print(f"  Anomali terdeteksi : {n_anomaly}  ({n_anomaly/len(labels_iso)*100:.1f}%)")
print(f"\n  Skor Anomali (decision_function):")
print(f"    Min  : {scores.min():.4f}")
print(f"    Maks : {scores.max():.4f}")
print(f"    Rata : {scores.mean():.4f}")
print(f"    Std  : {scores.std():.4f}")
print(f"\n  Catatan: nilai di bawah 0 → anomali; nilai di atas 0 → normal")

# Tampilkan beberapa sampel anomali teratas
anomaly_idx = np.argsort(scores)[:5]
print("\n  5 Sampel paling anomali:")
sample_anomaly = df_water[INPUT_COLS_WATER].iloc[anomaly_idx][
    ['COD_init_mgL', 'current_A', 'duration_min', 'voltage_V', 'electrode_gap_cm']
]
print(sample_anomaly.to_string(index=False))

save_model(iso_forest, 'isolation_forest.pkl')
save_model(scaler_iso, 'scaler_isolation.pkl')


# ============================================================
# ██  BAGIAN 2: ENERGY MANAGEMENT — PREPROCESSING
# ============================================================
print_section("BAGIAN 2 — ENERGY MANAGEMENT: PREPROCESSING")

df_e = df_energy.copy()

# --- Cyclical Encoding: hour ---
# Jam 23 dan jam 0 bersebelahan secara waktu, tapi angka 23 vs 0
# terlihat jauh oleh model. Solusi: proyeksi ke lingkaran unit.
df_e['hour_sin'] = np.sin(2 * np.pi * df_e['hour'] / 24)
df_e['hour_cos'] = np.cos(2 * np.pi * df_e['hour'] / 24)
df_e.drop(columns=['hour', 'time_minutes'], inplace=True)
print("  [+] hour → hour_sin, hour_cos")
print("  [-] Kolom 'hour' dan 'time_minutes' dihapus")

# --- Cyclical Encoding: wind_direction_deg ---
# Arah 0° dan 360° adalah identik. Tanpa encoding ini model menganggapnya
# sebagai nilai paling berjauhan.
df_e['wind_sin'] = np.sin(2 * np.pi * df_e['wind_direction_deg'] / 360)
df_e['wind_cos'] = np.cos(2 * np.pi * df_e['wind_direction_deg'] / 360)
df_e.drop(columns=['wind_direction_deg'], inplace=True)
print("  [+] wind_direction_deg → wind_sin, wind_cos")
print("  [-] Kolom 'wind_direction_deg' dihapus")

# --- Pisahkan fitur dan target ---
FEATURES_ENERGY = [c for c in df_e.columns if c not in TARGET_ENERGY]
X_energy = df_e[FEATURES_ENERGY]
y_energy = df_e[TARGET_ENERGY]

print(f"\n  Fitur setelah encoding : {len(FEATURES_ENERGY)} kolom")
print(f"    {FEATURES_ENERGY}")

# --- StandardScaler untuk semua fitur numerik ---
# RF tidak wajib, tapi scaling dilakukan sekarang supaya data siap
# jika nanti ingin dicoba SVR, Neural Network, atau model lain.
scaler_energy   = StandardScaler()
X_energy_scaled = scaler_energy.fit_transform(X_energy)
print("\n  [+] StandardScaler diterapkan ke semua fitur energi")

# --- Train/Test Split (tidak perlu stratified, target kontinu) ---
X_tr_en, X_te_en, y_tr_en, y_te_en = train_test_split(
    X_energy_scaled, y_energy,
    test_size=TEST_SIZE, random_state=RANDOM_STATE,
)
print(f"  Train: {len(X_tr_en)}  |  Test: {len(X_te_en)}")


# ============================================================
# ██  MODEL 4: Multi-output RF Regressor — Energy Management
# ============================================================
print_section("MODEL 4 — RF Regressor Multi-output: Energy Management")
print("  4 target diprediksi sekaligus (RF mendukung multi-output secara native)")

rf_energy = RandomForestRegressor(
    n_estimators=200,
    max_depth=None,
    min_samples_leaf=2,
    n_jobs=-1,
    random_state=RANDOM_STATE,
)
rf_energy.fit(X_tr_en, y_tr_en)

y_pred_en = rf_energy.predict(X_te_en)

print("\n  Metrik per target (Test Set):")
print(f"  {'Target':<28}  {'MAE':>9}  {'R²':>8}")
print(f"  {'-'*28}  {'-'*9}  {'-'*8}")
for i, col in enumerate(TARGET_ENERGY):
    mae = mean_absolute_error(y_te_en[col], y_pred_en[:, i])
    r2  = r2_score(y_te_en[col], y_pred_en[:, i])
    unit = '%' if 'fraction' in col else 'Wh'
    print(f"  {col:<28}  {mae:>7.3f}{unit}  {r2:>8.4f}")

# Feature importance untuk model energi
importances_e = pd.Series(rf_energy.feature_importances_, index=FEATURES_ENERGY)
print("\n  Top-5 Feature Importance:")
for feat, val in importances_e.nlargest(5).items():
    bar = '█' * int(val * 300)
    print(f"    {feat:<25} {val:.4f}  {bar}")

save_model(rf_energy, 'rf_regressor_energy.pkl')
save_model(scaler_energy, 'scaler_energy.pkl')


# ============================================================
# ██  RINGKASAN AKHIR
# ============================================================
print_section("TRAINING SELESAI — RINGKASAN FILE MODEL")

model_files = [
    ('rf_regressor_removal.pkl', 'RF Regressor  — 5 target penyisihan polutan (%)'),
    ('rf_classifier_clean.pkl',  'RF Classifier — label_clean 0/1 (balanced)'),
    ('isolation_forest.pkl',     'Isolation Forest — anomali 16 fitur input'),
    ('scaler_isolation.pkl',     'StandardScaler  — pasangan Isolation Forest'),
    ('rf_regressor_energy.pkl',  'RF Regressor  — 4 target energi (multi-output)'),
    ('scaler_energy.pkl',        'StandardScaler  — pasangan Energy RF'),
]

total_size = 0
for fname, desc in model_files:
    path = os.path.join(MODELS_DIR, fname)
    size = os.path.getsize(path) / 1024
    total_size += size
    print(f"  {fname:<35} {size:>8.1f} KB  |  {desc}")

print(f"\n  Total ukuran models/ : {total_size/1024:.2f} MB")
print(f"\n  Cara load model:")
print("    import pickle")
print("    with open('models/rf_regressor_removal.pkl', 'rb') as f:")
print("        model = pickle.load(f)")
print("    predictions = model.predict(X_new)  # X_new harus include Q_coulomb")