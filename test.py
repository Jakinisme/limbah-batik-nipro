"""
test_robustness.py
==================
Test suite untuk menguji ketangguhan model sistem penjernihan limbah batik.

Kategori test:
  A. Output Bounds      — apakah output selalu dalam rentang logis?
  B. Physics Monotonicity — apakah output konsisten dengan hukum fisika?
  C. Edge Cases         — nilai tepat di batas min/maks training range
  D. Out-of-Distribution— nilai di luar training range, model tidak boleh crash
  E. Anomaly Detection  — apakah Isolation Forest mendeteksi anomali yang jelas?
  F. Classifier Proba   — apakah probabilitas BERSIH berjalan sesuai arah yang benar?
  G. Energy Consistency — apakah fraksi solar+angin+baterai ≈ 100%?

Prasyarat:
  Jalankan generate_dataset.py dan train_models.py terlebih dahulu.
"""

import pickle
import numpy as np
import pandas as pd

# ============================================================
# KONFIGURASI
# ============================================================
MODELS_DIR      = 'models'
PASS_SYMBOL     = '✓  PASS'
FAIL_SYMBOL     = '✗  FAIL'
WARN_SYMBOL     = '⚠  WARN'

# Kolom fitur — harus sama persis dengan urutan saat training
WATER_FEATURE_COLS = [
    'pH_init', 'turbidity_init_NTU', 'TDS_init_mgL', 'conductivity_uScm',
    'color_init_PtCo', 'COD_init_mgL', 'BOD_init_mgL', 'temp_water_C',
    'TSS_init_mgL', 'voltage_V', 'current_A', 'duration_min',
    'electrode_type', 'electrode_gap_cm', 'power_consumption_W',
    'inflow_rate_Lmin', 'Q_coulomb',
]

ISO_FEATURE_COLS = [
    'pH_init', 'turbidity_init_NTU', 'TDS_init_mgL', 'conductivity_uScm',
    'color_init_PtCo', 'COD_init_mgL', 'BOD_init_mgL', 'temp_water_C',
    'TSS_init_mgL', 'voltage_V', 'current_A', 'duration_min',
    'electrode_type', 'electrode_gap_cm', 'power_consumption_W',
    'inflow_rate_Lmin',
]

ENERGY_FEATURE_COLS = [
    'irradiance_Wm2', 'solar_output_W', 'wind_speed_ms', 'wind_output_W',
    'battery_SOC_pct', 'air_temp_C', 'ec_power_demand_W',
    'hour_sin', 'hour_cos', 'wind_sin', 'wind_cos',
]

TARGET_REMOVAL = [
    'COD_removal_pct', 'BOD_removal_pct', 'TSS_removal_pct',
    'color_removal_pct', 'turbidity_removal_pct',
]

TARGET_ENERGY = [
    'solar_fraction_pct', 'wind_fraction_pct',
    'battery_fraction_pct', 'energy_forecast_1h_Wh',
]

# ============================================================
# HELPER: LOAD MODEL
# ============================================================
def load_model(filename):
    with open(f'{MODELS_DIR}/{filename}', 'rb') as f:
        return pickle.load(f)

# ============================================================
# HELPER: BUAT INPUT WATER (otomatis hitung Q_coulomb)
# ============================================================
def make_water_df(
    pH_init=8.5, turbidity_init_NTU=3000.0, TDS_init_mgL=3000.0,
    conductivity_uScm=8000.0, color_init_PtCo=800.0, COD_init_mgL=3000.0,
    BOD_init_mgL=900.0, temp_water_C=28.0, TSS_init_mgL=1500.0,
    voltage_V=14.0, current_A=5.0, duration_min=45,
    electrode_type=0, electrode_gap_cm=2.0,
    power_consumption_W=None, inflow_rate_Lmin=25.0,
):
    """Buat satu baris input water dengan Q_coulomb otomatis."""
    if power_consumption_W is None:
        power_consumption_W = voltage_V * current_A
    Q_coulomb = current_A * duration_min * 60

    return pd.DataFrame([{
        'pH_init'              : pH_init,
        'turbidity_init_NTU'   : turbidity_init_NTU,
        'TDS_init_mgL'         : TDS_init_mgL,
        'conductivity_uScm'    : conductivity_uScm,
        'color_init_PtCo'      : color_init_PtCo,
        'COD_init_mgL'         : COD_init_mgL,
        'BOD_init_mgL'         : BOD_init_mgL,
        'temp_water_C'         : temp_water_C,
        'TSS_init_mgL'         : TSS_init_mgL,
        'voltage_V'            : voltage_V,
        'current_A'            : current_A,
        'duration_min'         : duration_min,
        'electrode_type'       : electrode_type,
        'electrode_gap_cm'     : electrode_gap_cm,
        'power_consumption_W'  : power_consumption_W,
        'inflow_rate_Lmin'     : inflow_rate_Lmin,
        'Q_coulomb'            : Q_coulomb,
    }])[WATER_FEATURE_COLS]


def make_iso_df(**kwargs):
    """Buat input Isolation Forest (tanpa Q_coulomb)."""
    df = make_water_df(**kwargs)
    return df[ISO_FEATURE_COLS]


# ============================================================
# HELPER: BUAT INPUT ENERGY (otomatis cyclical encoding)
# ============================================================
def make_energy_df(
    irradiance_Wm2=600.0, solar_output_W=45.0,
    wind_speed_ms=4.0, wind_output_W=10.0,
    battery_SOC_pct=70.0, air_temp_C=28.0,
    ec_power_demand_W=80.0,
    hour=12, wind_direction_deg=90.0,
):
    """Buat satu baris input energi dengan cyclical encoding otomatis."""
    return pd.DataFrame([{
        'irradiance_Wm2'   : irradiance_Wm2,
        'solar_output_W'   : solar_output_W,
        'wind_speed_ms'    : wind_speed_ms,
        'wind_output_W'    : wind_output_W,
        'battery_SOC_pct'  : battery_SOC_pct,
        'air_temp_C'       : air_temp_C,
        'ec_power_demand_W': ec_power_demand_W,
        'hour_sin'         : np.sin(2 * np.pi * hour / 24),
        'hour_cos'         : np.cos(2 * np.pi * hour / 24),
        'wind_sin'         : np.sin(2 * np.pi * wind_direction_deg / 360),
        'wind_cos'         : np.cos(2 * np.pi * wind_direction_deg / 360),
    }])[ENERGY_FEATURE_COLS]


# ============================================================
# HELPER: UTILITAS TEST
# ============================================================
results = []

def check(name, condition, detail='', warn=False):
    """Catat satu hasil test dan print ke layar."""
    if warn:
        symbol = WARN_SYMBOL
        status = 'WARN'
    elif condition:
        symbol = PASS_SYMBOL
        status = 'PASS'
    else:
        symbol = FAIL_SYMBOL
        status = 'FAIL'

    results.append({'name': name, 'status': status, 'detail': detail})
    detail_str = f'  → {detail}' if detail else ''
    print(f'  {symbol} │ {name}{detail_str}')


def section(title):
    print()
    print('┌' + '─' * 62 + '┐')
    print(f'│  {title:<60}│')
    print('└' + '─' * 62 + '┘')


# ============================================================
# LOAD SEMUA MODEL
# ============================================================
print()
print('=' * 64)
print('  LOADING MODELS')
print('=' * 64)

rf_reg    = load_model('rf_regressor_removal.pkl')
rf_clf    = load_model('rf_classifier_clean.pkl')
iso       = load_model('isolation_forest.pkl')
scaler_iso = load_model('scaler_isolation.pkl')
rf_energy = load_model('rf_regressor_energy.pkl')
scaler_e  = load_model('scaler_energy.pkl')

print('  Semua model berhasil dimuat.')


# ============================================================
# A. OUTPUT BOUNDS
# ============================================================
section('A. OUTPUT BOUNDS — Output selalu dalam rentang logis?')

# A-1: Removal % harus 0–100 untuk input acak
rng_test = np.random.RandomState(999)
n_rand   = 200
rand_inputs = pd.DataFrame({
    'pH_init'             : rng_test.uniform(7.5, 11.0, n_rand),
    'turbidity_init_NTU'  : rng_test.uniform(500, 9500, n_rand),
    'TDS_init_mgL'        : rng_test.uniform(500, 9500, n_rand),
    'conductivity_uScm'   : rng_test.uniform(3000, 15000, n_rand),
    'color_init_PtCo'     : rng_test.uniform(200, 2000, n_rand),
    'COD_init_mgL'        : rng_test.uniform(300, 21000, n_rand),
    'BOD_init_mgL'        : rng_test.uniform(75, 11550, n_rand),
    'temp_water_C'        : rng_test.uniform(20, 35, n_rand),
    'TSS_init_mgL'        : rng_test.uniform(200, 7000, n_rand),
    'voltage_V'           : rng_test.uniform(8, 20, n_rand),
    'current_A'           : rng_test.uniform(0.5, 10, n_rand),
    'duration_min'        : rng_test.randint(15, 91, n_rand).astype(float),
    'electrode_type'      : rng_test.choice([0, 1], n_rand),
    'electrode_gap_cm'    : rng_test.uniform(1, 5, n_rand),
    'power_consumption_W' : rng_test.uniform(4, 200, n_rand),
    'inflow_rate_Lmin'    : rng_test.uniform(5, 50, n_rand),
})[ISO_FEATURE_COLS]  # 16 kolom dulu, lalu tambah Q
rand_inputs['Q_coulomb'] = rand_inputs['current_A'] * rand_inputs['duration_min'] * 60
rand_inputs = rand_inputs[WATER_FEATURE_COLS]

preds_removal = rf_reg.predict(rand_inputs)
all_in_bounds = np.all((preds_removal >= 0) & (preds_removal <= 100))
viol = (~((preds_removal >= 0) & (preds_removal <= 100))).sum()
check(
    f'[A-1] Removal % dalam 0–100% ({n_rand} sampel acak)',
    all_in_bounds,
    f'{viol} pelanggaran dari {n_rand * len(TARGET_REMOVAL)} prediksi'
    if not all_in_bounds else f'min={preds_removal.min():.1f}%  max={preds_removal.max():.1f}%'
)

# A-2: Probabilitas classifier harus 0–1
prob_clf = rf_clf.predict_proba(rand_inputs)
check(
    f'[A-2] Probabilitas classifier dalam [0, 1] ({n_rand} sampel)',
    np.all((prob_clf >= 0) & (prob_clf <= 1)),
    f'min={prob_clf.min():.4f}  max={prob_clf.max():.4f}'
)

# A-3: Sum probabilitas kelas = 1 (harusnya selalu, tapi cek saja)
prob_sum = prob_clf.sum(axis=1)
check(
    '[A-3] Sum prob kelas = 1.0 (constraint probabilistik)',
    np.allclose(prob_sum, 1.0, atol=1e-6),
    f'max_dev={np.abs(prob_sum - 1.0).max():.2e}'
)

# A-4: Energy fraction individual dalam 0–100
rand_e_rows = []
for h in range(0, 24, 2):
    ws = rng_test.uniform(0, 10)
    irr = max(0, np.sin(np.pi * (h - 6) / 12)) * rng_test.uniform(600, 1320)
    rand_e_rows.append(make_energy_df(
        irradiance_Wm2=irr,
        solar_output_W=min(irr * 0.15, 100),
        wind_speed_ms=ws,
        wind_output_W=min((ws / 13) ** 3 * 300, 300) if ws >= 2 else 0,
        battery_SOC_pct=rng_test.uniform(20, 100),
        ec_power_demand_W=rng_test.uniform(10, 160),
        hour=h,
    ))
rand_e = pd.concat(rand_e_rows, ignore_index=True)
rand_e_scaled = scaler_e.transform(rand_e)
preds_e = rf_energy.predict(rand_e_scaled)
df_preds_e = pd.DataFrame(preds_e, columns=TARGET_ENERGY)

check(
    '[A-4] Fraksi energi individual dalam [0, 100]%',
    np.all(
        (df_preds_e[['solar_fraction_pct','wind_fraction_pct','battery_fraction_pct']] >= 0).values &
        (df_preds_e[['solar_fraction_pct','wind_fraction_pct','battery_fraction_pct']] <= 100).values
    ),
    f"solar:[{df_preds_e['solar_fraction_pct'].min():.1f},{df_preds_e['solar_fraction_pct'].max():.1f}]  "
    f"wind:[{df_preds_e['wind_fraction_pct'].min():.1f},{df_preds_e['wind_fraction_pct'].max():.1f}]  "
    f"batt:[{df_preds_e['battery_fraction_pct'].min():.1f},{df_preds_e['battery_fraction_pct'].max():.1f}]"
)

# A-5: Energy forecast tidak boleh negatif
check(
    '[A-5] Energy forecast ≥ 0 Wh',
    (df_preds_e['energy_forecast_1h_Wh'] >= 0).all(),
    f"min={df_preds_e['energy_forecast_1h_Wh'].min():.2f} Wh"
)


# ============================================================
# B. PHYSICS MONOTONICITY
# ============================================================
section('B. PHYSICS MONOTONICITY — Output konsisten dengan hukum fisika?')

# B-1: Makin tinggi Q → COD_removal harus ≥ sebelumnya
# Variasikan Q dengan menaikkan current, kondisi lain tetap
Q_levels = [
    (0.5, 15),   # Q sangat kecil: 450 C
    (2.0, 30),   # Q kecil      : 3600 C
    (5.0, 45),   # Q sedang     : 13500 C
    (7.0, 60),   # Q besar      : 25200 C
    (10.0, 90),  # Q maksimal   : 54000 C
]
cod_removals = []
for cur, dur in Q_levels:
    df_q = make_water_df(current_A=cur, duration_min=dur,
                         COD_init_mgL=3000, pH_init=8.0,
                         electrode_type=0, electrode_gap_cm=2.0)
    pred = rf_reg.predict(df_q)
    cod_removals.append(pred[0, 0])

is_monotone = all(
    cod_removals[i] <= cod_removals[i + 1] + 2.0  # toleransi 2% untuk noise RF
    for i in range(len(cod_removals) - 1)
)
q_str = '  '.join([f'Q={cur*dur*60:.0f}C→{r:.1f}%' for (cur, dur), r in zip(Q_levels, cod_removals)])
check(
    '[B-1] COD removal meningkat seiring Q_coulomb naik (monotonicity)',
    is_monotone,
    q_str
)

# B-2: Elektroda Al (0) ≥ elektroda Fe (1) untuk COD removal
#      (sesuai bonus di model training: Al +2%, Fe +0%)
df_al = make_water_df(electrode_type=0, current_A=5, duration_min=45,
                      COD_init_mgL=3000, pH_init=8.0)
df_fe = make_water_df(electrode_type=1, current_A=5, duration_min=45,
                      COD_init_mgL=3000, pH_init=8.0)
rem_al = rf_reg.predict(df_al)[0, 0]
rem_fe = rf_reg.predict(df_fe)[0, 0]
check(
    '[B-2] Al electrode removal ≥ Fe electrode (semua kondisi lain sama)',
    rem_al >= rem_fe - 1.0,
    f'Al={rem_al:.2f}%  Fe={rem_fe:.2f}%  Δ={rem_al - rem_fe:+.2f}%'
)

# B-3: Jarak elektroda lebih besar → removal turun
df_gap_small = make_water_df(electrode_gap_cm=1.0, current_A=5, duration_min=45)
df_gap_large = make_water_df(electrode_gap_cm=5.0, current_A=5, duration_min=45)
rem_small_gap = rf_reg.predict(df_gap_small)[0, 0]
rem_large_gap = rf_reg.predict(df_gap_large)[0, 0]
check(
    '[B-3] Gap kecil (1cm) removal ≥ gap besar (5cm)',
    rem_small_gap >= rem_large_gap - 1.0,
    f'gap=1cm:{rem_small_gap:.2f}%  gap=5cm:{rem_large_gap:.2f}%  Δ={rem_small_gap - rem_large_gap:+.2f}%'
)

# B-4: Solar fraction lebih tinggi di siang hari (irradiance tinggi)
df_noon  = make_energy_df(hour=12, irradiance_Wm2=1200, solar_output_W=90,
                           wind_speed_ms=1.0, wind_output_W=0.0, battery_SOC_pct=60)
df_night = make_energy_df(hour=0, irradiance_Wm2=0, solar_output_W=0,
                           wind_speed_ms=1.0, wind_output_W=0.0, battery_SOC_pct=60)
solar_noon  = rf_energy.predict(scaler_e.transform(df_noon))[0, 0]
solar_night = rf_energy.predict(scaler_e.transform(df_night))[0, 0]
check(
    '[B-4] Solar fraction: siang (irr=1200) > malam (irr=0)',
    solar_noon > solar_night,
    f'siang={solar_noon:.1f}%  malam={solar_night:.1f}%'
)

# B-5: Wind fraction lebih tinggi saat angin kencang
df_wind_low  = make_energy_df(wind_speed_ms=1.0, wind_output_W=0.0,
                               irradiance_Wm2=0, solar_output_W=0, hour=2)
df_wind_high = make_energy_df(wind_speed_ms=10.0, wind_output_W=170.0,
                               irradiance_Wm2=0, solar_output_W=0, hour=2)
wf_low  = rf_energy.predict(scaler_e.transform(df_wind_low))[0, 1]
wf_high = rf_energy.predict(scaler_e.transform(df_wind_high))[0, 1]
check(
    '[B-5] Wind fraction: angin kencang (10 m/s) > angin lemah (1 m/s)',
    wf_high > wf_low,
    f'10m/s={wf_high:.1f}%  1m/s={wf_low:.1f}%'
)


# ============================================================
# C. EDGE CASES — Nilai tepat di batas training range
# ============================================================
section('C. EDGE CASES — Nilai tepat di batas min/maks training range')

# C-1: Nilai minimum semua parameter
df_min = make_water_df(
    pH_init=7.5, turbidity_init_NTU=500, TDS_init_mgL=500,
    conductivity_uScm=3000, color_init_PtCo=200, COD_init_mgL=300,
    BOD_init_mgL=75, temp_water_C=20, TSS_init_mgL=200,
    voltage_V=8, current_A=0.5, duration_min=15,
    electrode_type=0, electrode_gap_cm=1.0, inflow_rate_Lmin=5.0,
)
try:
    pred_min = rf_reg.predict(df_min)
    ok = np.all((pred_min >= 0) & (pred_min <= 100))
    check('[C-1] Input MINIMUM → tidak crash, output in-bounds',
          ok, f'COD_removal={pred_min[0, 0]:.1f}%')
except Exception as ex:
    check('[C-1] Input MINIMUM → tidak crash, output in-bounds', False, str(ex))

# C-2: Nilai maksimum semua parameter
df_max = make_water_df(
    pH_init=11.0, turbidity_init_NTU=9500, TDS_init_mgL=9500,
    conductivity_uScm=15000, color_init_PtCo=2000, COD_init_mgL=21000,
    BOD_init_mgL=11550, temp_water_C=35, TSS_init_mgL=7000,
    voltage_V=20, current_A=10, duration_min=90,
    electrode_type=1, electrode_gap_cm=5.0, inflow_rate_Lmin=50.0,
)
try:
    pred_max = rf_reg.predict(df_max)
    ok = np.all((pred_max >= 0) & (pred_max <= 100))
    check('[C-2] Input MAKSIMUM → tidak crash, output in-bounds',
          ok, f'COD_removal={pred_max[0, 0]:.1f}%')
except Exception as ex:
    check('[C-2] Input MAKSIMUM → tidak crash, output in-bounds', False, str(ex))

# C-3: Battery SOC minimum (20%)
df_lowbatt = make_energy_df(battery_SOC_pct=20, irradiance_Wm2=0,
                             solar_output_W=0, wind_speed_ms=1, wind_output_W=0, hour=2)
try:
    pred_lb = rf_energy.predict(scaler_e.transform(df_lowbatt))
    check('[C-3] Battery SOC=20% (minimum) → tidak crash',
          True, f'battery_frac={pred_lb[0, 2]:.1f}%')
except Exception as ex:
    check('[C-3] Battery SOC=20% (minimum) → tidak crash', False, str(ex))

# C-4: Battery SOC maksimum (100%)
df_fullbatt = make_energy_df(battery_SOC_pct=100, irradiance_Wm2=0,
                              solar_output_W=0, wind_speed_ms=1, wind_output_W=0, hour=2)
try:
    pred_fb = rf_energy.predict(scaler_e.transform(df_fullbatt))
    check('[C-4] Battery SOC=100% (maksimum) → tidak crash',
          True, f'battery_frac={pred_fb[0, 2]:.1f}%')
except Exception as ex:
    check('[C-4] Battery SOC=100% (maksimum) → tidak crash', False, str(ex))


# ============================================================
# D. OUT-OF-DISTRIBUTION (OOD)
# ============================================================
section('D. OUT-OF-DISTRIBUTION — Nilai di luar training range')
print('  (OOD tidak harus akurat, tapi tidak boleh crash atau return NaN/Inf)')

ood_cases = [
    # (label, kwargs_overrides, tipe)
    ('[D-1] COD sangat ekstrem (50000 mg/L — 2.4× maks training)',
     dict(COD_init_mgL=50000), 'water'),
    ('[D-2] Arus negatif (-1A — fisika tidak mungkin)',
     dict(current_A=-1.0, duration_min=45), 'water'),
    ('[D-3] pH ekstrem asam (4.0 — di bawah training 7.5)',
     dict(pH_init=4.0), 'water'),
    ('[D-4] Durasi sangat panjang (200 menit — 2.2× maks training)',
     dict(current_A=5, duration_min=200), 'water'),
    ('[D-5] Irradiance sangat tinggi (2000 W/m² — tropical maximum)',
     dict(irradiance_Wm2=2000, solar_output_W=100, hour=12), 'energy'),
    ('[D-6] Wind speed sangat kencang (30 m/s — badai)',
     dict(wind_speed_ms=30, wind_output_W=300, hour=2), 'energy'),
]

for label, kwargs, mode in ood_cases:
    try:
        if mode == 'water':
            df_ood = make_water_df(**kwargs)
            pred   = rf_reg.predict(df_ood)
            has_nan_inf = np.any(~np.isfinite(pred))
            check(label, not has_nan_inf,
                  f'COD_removal={pred[0,0]:.1f}% — {"NaN/Inf!" if has_nan_inf else "finite OK"}')
        else:
            df_ood  = make_energy_df(**kwargs)
            scaled  = scaler_e.transform(df_ood)
            pred    = rf_energy.predict(scaled)
            has_nan_inf = np.any(~np.isfinite(pred))
            check(label, not has_nan_inf,
                  f'solar={pred[0,0]:.1f}%  wind={pred[0,1]:.1f}% — {"NaN/Inf!" if has_nan_inf else "finite OK"}')
    except Exception as ex:
        check(label, False, f'EXCEPTION: {ex}')


# ============================================================
# E. ANOMALY DETECTION
# ============================================================
section('E. ANOMALY DETECTION — Isolation Forest mendeteksi anomali nyata?')

def predict_iso(df_16col):
    scaled = scaler_iso.transform(df_16col[ISO_FEATURE_COLS])
    label  = iso.predict(scaled)[0]   # -1=anomali, 1=normal
    score  = iso.decision_function(scaled)[0]
    return label, score

# E-1: Operasi normal → harus NORMAL
df_normal = make_water_df(
    COD_init_mgL=3000, current_A=5.0, duration_min=45,
    pH_init=8.5, electrode_gap_cm=2.0, voltage_V=14,
)
lbl, sc = predict_iso(df_normal)
check('[E-1] Operasi normal → NORMAL (+1)',
      lbl == 1, f'label={lbl}  score={sc:.4f}')

# E-2: COD sangat tinggi + arus sangat rendah + gap besar → ANOMALI
# (limbah pekat tapi daya proses sangat lemah — tidak masuk akal operasional)
df_anom1 = make_water_df(
    COD_init_mgL=20000, current_A=0.5, duration_min=15,
    electrode_gap_cm=5.0, voltage_V=8,
)
lbl, sc = predict_iso(df_anom1)
check('[E-2] COD_init=20000 + arus=0.5A + gap=5cm → ANOMALI (-1)',
      lbl == -1, f'label={lbl}  score={sc:.4f}')

# E-3: pH sangat ekstrem alkali (11) + konduktivitas sangat tinggi → ANOMALI
df_anom2 = make_water_df(
    pH_init=11.0, conductivity_uScm=15000,
    COD_init_mgL=20000, TSS_init_mgL=7000,
)
lbl, sc = predict_iso(df_anom2)
check('[E-3] pH=11 + konduktivitas=15000 + COD=20000 + TSS=7000 → ANOMALI',
      lbl == -1, f'label={lbl}  score={sc:.4f}',
      warn=(lbl != -1))

# E-4: Nilai semua fitur di median → harus NORMAL
df_median = make_water_df(
    pH_init=9.25, turbidity_init_NTU=5000, TDS_init_mgL=5000,
    conductivity_uScm=9000, color_init_PtCo=1100, COD_init_mgL=3000,
    BOD_init_mgL=750, temp_water_C=27.5, TSS_init_mgL=1000,
    voltage_V=14, current_A=5.25, duration_min=52,
    electrode_type=0, electrode_gap_cm=3.0, inflow_rate_Lmin=27.5,
)
lbl, sc = predict_iso(df_median)
check('[E-4] Semua fitur di nilai median training → NORMAL (+1)',
      lbl == 1, f'label={lbl}  score={sc:.4f}')

# E-5: Beberapa sampel normal → anomali rate harus < 30%
rows_normal = [make_water_df(
    COD_init_mgL=float(c), current_A=5.0, duration_min=45
) for c in [500, 1000, 2000, 3000, 5000, 7000, 10000, 15000]]

preds_batch = [predict_iso(r)[0] for r in rows_normal]
anom_rate = preds_batch.count(-1) / len(preds_batch)
check('[E-5] 8 variasi COD normal → anomali rate < 30%',
      anom_rate < 0.30,
      f'{preds_batch.count(-1)}/{len(preds_batch)} anomali ({anom_rate*100:.0f}%)')


# ============================================================
# F. CLASSIFIER PROBABILITY ORDERING
# ============================================================
section('F. CLASSIFIER PROBABILITY — Probabilitas BERSIH searah kondisi air')

def bersih_prob(df):
    return rf_clf.predict_proba(df)[0, 1]

# Kondisi optimal: COD rendah, arus tinggi, durasi panjang, Al electrode, gap kecil
df_optimal = make_water_df(
    COD_init_mgL=300, current_A=10, duration_min=90,
    electrode_type=0, electrode_gap_cm=1.0, pH_init=7.5,
    turbidity_init_NTU=600, TDS_init_mgL=600, TSS_init_mgL=250,
)
prob_optimal = bersih_prob(df_optimal)

# Kondisi buruk: COD sangat tinggi, arus rendah, durasi singkat
df_bad = make_water_df(
    COD_init_mgL=18000, current_A=0.5, duration_min=15,
    electrode_type=1, electrode_gap_cm=5.0, pH_init=10.5,
    turbidity_init_NTU=9000, TDS_init_mgL=9000, TSS_init_mgL=6000,
)
prob_bad = bersih_prob(df_bad)

# Kondisi sedang: parameter tengah-tengah
df_mid = make_water_df(
    COD_init_mgL=3000, current_A=5, duration_min=45,
    electrode_type=0, electrode_gap_cm=2.5, pH_init=8.5,
)
prob_mid = bersih_prob(df_mid)

check('[F-1] P(BERSIH | optimal) > P(BERSIH | sedang) > P(BERSIH | buruk)',
      prob_optimal > prob_mid > prob_bad,
      f'optimal={prob_optimal:.4f}  sedang={prob_mid:.4f}  buruk={prob_bad:.4f}')

check('[F-2] P(BERSIH | kondisi buruk) < 0.3',
      prob_bad < 0.3,
      f'prob_buruk={prob_bad:.4f}')

check('[F-3] P(BERSIH | kondisi optimal) > 0.3',
      prob_optimal > 0.3,
      f'prob_optimal={prob_optimal:.4f}')

# F-4: Test monotonicity COD terhadap probabilitas BERSIH
cod_vals   = [300, 500, 1000, 2000, 5000, 10000, 20000]
probs_cod  = [bersih_prob(make_water_df(COD_init_mgL=c, current_A=10, duration_min=90)) for c in cod_vals]
is_monotone_prob = all(
    probs_cod[i] >= probs_cod[i + 1] - 0.05  # toleransi kecil
    for i in range(len(probs_cod) - 1)
)
prob_str = '  '.join([f'COD={c}→{p:.3f}' for c, p in zip(cod_vals, probs_cod)])
check('[F-4] P(BERSIH) turun saat COD_init naik (monotonicity)',
      is_monotone_prob, prob_str)


# ============================================================
# G. ENERGY CONSISTENCY
# ============================================================
section('G. ENERGY CONSISTENCY — Fraksi alokasi energi logis?')

# G-1: Siang hari cerah, angin tenang → solar fraction harus dominan
df_sunny = make_energy_df(
    hour=12, irradiance_Wm2=1200, solar_output_W=90,
    wind_speed_ms=1.0, wind_output_W=0.0, battery_SOC_pct=60,
    ec_power_demand_W=80,
)
pred_sunny = rf_energy.predict(scaler_e.transform(df_sunny))[0]
sf_sunny, wf_sunny, bf_sunny, _ = pred_sunny
check('[G-1] Siang cerah (irr=1200) → solar fraction dominan (> 50%)',
      sf_sunny > 50,
      f'solar={sf_sunny:.1f}%  wind={wf_sunny:.1f}%  batt={bf_sunny:.1f}%')

# G-2: Malam hari, angin kencang → wind fraction dominan
df_windy_night = make_energy_df(
    hour=2, irradiance_Wm2=0, solar_output_W=0,
    wind_speed_ms=12.0, wind_output_W=280.0, battery_SOC_pct=60,
    ec_power_demand_W=80,
)
pred_wn = rf_energy.predict(scaler_e.transform(df_windy_night))[0]
sf_wn, wf_wn, bf_wn, _ = pred_wn
check('[G-2] Malam, angin kencang (12 m/s) → wind fraction dominan (> 50%)',
      wf_wn > 50,
      f'solar={sf_wn:.1f}%  wind={wf_wn:.1f}%  batt={bf_wn:.1f}%')

# G-3: Tidak ada sumber EBT sama sekali → battery fraction dominan
df_no_ren = make_energy_df(
    hour=2, irradiance_Wm2=0, solar_output_W=0,
    wind_speed_ms=0.5, wind_output_W=0.0, battery_SOC_pct=80,
    ec_power_demand_W=60,
)
pred_nr = rf_energy.predict(scaler_e.transform(df_no_ren))[0]
sf_nr, wf_nr, bf_nr, _ = pred_nr
check('[G-3] Tidak ada solar/angin → battery fraction dominan (> 50%)',
      bf_nr > 50,
      f'solar={sf_nr:.1f}%  wind={wf_nr:.1f}%  batt={bf_nr:.1f}%')

# G-4: Forecast tidak boleh nol saat ada sumber energi
df_has_energy = make_energy_df(
    hour=12, irradiance_Wm2=800, solar_output_W=60,
    wind_speed_ms=5.0, wind_output_W=30.0,
)
pred_fcast = rf_energy.predict(scaler_e.transform(df_has_energy))[0, 3]
check('[G-4] Forecast > 0 Wh saat ada solar + angin tersedia',
      pred_fcast > 0,
      f'energy_forecast={pred_fcast:.2f} Wh')

# G-5: Forecast malam tanpa angin → lebih kecil dari siang hari cerah
df_noon_forecast  = make_energy_df(hour=12, irradiance_Wm2=1200, solar_output_W=90,
                                    wind_speed_ms=5, wind_output_W=30)
df_night_forecast = make_energy_df(hour=2, irradiance_Wm2=0, solar_output_W=0,
                                    wind_speed_ms=0.5, wind_output_W=0)
fcast_noon  = rf_energy.predict(scaler_e.transform(df_noon_forecast))[0, 3]
fcast_night = rf_energy.predict(scaler_e.transform(df_night_forecast))[0, 3]
check('[G-5] Forecast siang (solar+angin) > forecast malam (tanpa EBT)',
      fcast_noon > fcast_night,
      f'siang={fcast_noon:.2f} Wh  malam={fcast_night:.2f} Wh')


# ============================================================
# RINGKASAN AKHIR
# ============================================================
print()
print('=' * 64)
print('  RINGKASAN HASIL TEST')
print('=' * 64)

total  = len(results)
passed = sum(1 for r in results if r['status'] == 'PASS')
failed = sum(1 for r in results if r['status'] == 'FAIL')
warned = sum(1 for r in results if r['status'] == 'WARN')

print(f'  Total  : {total}')
print(f'  ✓ PASS : {passed}')
print(f'  ✗ FAIL : {failed}')
print(f'  ⚠ WARN : {warned}')

if failed > 0:
    print('\n  Test yang GAGAL:')
    for r in results:
        if r['status'] == 'FAIL':
            print(f'    ✗  {r["name"]}')
            if r['detail']:
                print(f'       {r["detail"]}')

score_pct = (passed / total) * 100
print(f'\n  Skor ketangguhan: {passed}/{total} ({score_pct:.0f}%)')

if score_pct == 100:
    print('  → Semua model lolos. Siap digunakan.')
elif score_pct >= 80:
    print('  → Model cukup robust. Periksa test yang WARN/FAIL.')
else:
    print('  → Banyak test gagal. Perlu investigasi lebih lanjut.')