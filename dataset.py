"""
generate_dataset.py
===================
Script generate data sintetis untuk sistem penjernihan limbah batik.

Menghasilkan dua dataset:
  1. dataset_water_quality.csv     → untuk Random Forest + Isolation Forest
  2. dataset_energy_management.csv → untuk model manajemen energi hibrida

Semua nilai berbasis range dari studi limbah batik Indonesia dan
persamaan fisika elektrokoagulasi (Hukum Faraday).

Referensi baku mutu: Permen LHK No. 16 Tahun 2019
"""

import numpy as np
import pandas as pd

np.random.seed(42)

# ============================================================
# KONSTANTA FISIKA
# ============================================================
FARADAY    = 96485   # C/mol
M_Al, n_Al = 27.0, 3  # Aluminium: Al³⁺
M_Fe, n_Fe = 56.0, 2  # Besi: Fe²⁺
BATCH_VOL  = 50.0    # Liter (volume batch pengolahan)
Q_REF      = 1000.0  # Coulomb (referensi scaling model)


# ============================================================
# FUNGSI GENERATE SAMPEL
# ============================================================
def generate_water_samples(n, cod_range, current_range, duration_range, seed_offset=0):
    """
    Generate n sampel data kualitas air + proses elektrokoagulasi.

    Parameter:
      n              : jumlah sampel
      cod_range      : (min, max) COD awal dalam mg/L
      current_range  : (min, max) arus elektroda dalam Ampere
      duration_range : (min, max) durasi proses dalam menit
      seed_offset    : offset random seed agar batch berbeda hasilnya
    """
    rng = np.random.RandomState(42 + seed_offset)

    # ---- Input: sensor kualitas awal limbah ----
    # Range dari studi batik Indonesia:
    #   pH      : 7.5–11 (alkali, umumnya 8–10)
    #   Turb    : 500–9500 NTU (Jetis home industry: 1000–9500)
    #   TDS     : 500–9500 mg/L (Sidoarjo: ~9150)
    #   Kondukt : 3000–15000 μS/cm (~11630 μS/cm)
    #   Warna   : 200–2000 Pt-Co
    #   COD     : 300–21000 mg/L (Jambi: 21233, Sidoarjo: 4163)
    #   BOD     : 25–55% dari COD
    #   TSS     : 200–7000 mg/L (Jetis: 200–3500, Jambi: 6900)
    pH_init        = rng.uniform(7.5, 11.0, n)
    turbidity_init = rng.uniform(500, 9500, n)
    TDS_init       = rng.uniform(500, 9500, n)
    conductivity   = rng.uniform(3000, 15000, n)
    color_init     = rng.uniform(200, 2000, n)
    COD_init       = np.exp(rng.uniform(np.log(cod_range[0]), np.log(cod_range[1]), n))
    BOD_init       = COD_init * rng.uniform(0.25, 0.55, n)
    temp_water     = rng.uniform(20, 35, n)
    TSS_init       = np.exp(rng.uniform(np.log(200), np.log(7000), n))

    # ---- Input: parameter proses elektrokoagulasi ----
    # Range dari eksperimen literatur Indonesia:
    #   Tegangan : 8–20 V (variasi 8, 12, 16, 20 V)
    #   Arus     : 0.5–10 A
    #   Durasi   : 15–90 menit
    #   Elektroda: Al (0) atau Fe (1)
    #   Jarak    : 1–5 cm (umumnya 2–5 cm)
    voltage        = rng.uniform(8, 20, n)
    current        = rng.uniform(current_range[0], current_range[1], n)
    duration       = rng.uniform(duration_range[0], duration_range[1], n)
    electrode_type = rng.choice([0, 1], n)   # 0 = Aluminium, 1 = Besi
    electrode_gap  = rng.uniform(1.0, 5.0, n)

    # ---- Model Fisika: Hukum Faraday ----
    # Muatan listrik total yang melewati elektroda (Coulomb)
    Q = current * duration * 60

    # Massa elektroda yang larut dan menjadi ion koagulan (gram)
    mass_Al        = Q * M_Al / (n_Al * FARADAY)
    mass_Fe        = Q * M_Fe / (n_Fe * FARADAY)
    mass_dissolved = np.where(electrode_type == 0, mass_Al, mass_Fe)
    dose           = mass_dissolved / BATCH_VOL   # g/L

    # ---- Hitung efisiensi penyisihan ----
    # Model logaritmik (diminishing returns):
    #   Q tipikal (5A × 45min = 13500 C) → base ~59%
    #   Q maksimal (10A × 90min = 54000 C) → base ~88%
    base_removal  = 22.0 * np.log1p(Q / Q_REF)

    # Koreksi pH: koagulasi optimal di pH 7.5, turun di pH ekstrem
    pH_correction = -2.0 * np.abs(pH_init - 7.5) / 3.5

    # Penalti konsentrasi: limbah sangat pekat butuh lebih banyak koagulan
    conc_pen      = -1.0 * np.log1p(COD_init / 10000)

    # Bonus jenis elektroda: Al sedikit lebih efektif untuk COD/warna
    elec_bonus    = np.where(electrode_type == 0, 2.0, 0.0)

    # Penalti jarak elektroda: gap besar → hambatan lebih tinggi         
    gap_pen       = -0.5 * (electrode_gap - 2.0)

    # COD removal (%) — basis utama
    COD_removal   = np.clip(
        base_removal + pH_correction + conc_pen + elec_bonus + gap_pen
        + rng.normal(0, 5, n),
        30, 92
    )

    # Removal lain diturunkan secara konsisten dari COD_removal
    TSS_removal   = np.clip(COD_removal + rng.uniform(3, 10, n),  40, 95)
    BOD_removal   = np.clip(COD_removal + rng.uniform(-3, 5, n),  30, 93)
    color_removal = np.clip(COD_removal + rng.uniform(5, 15, n),  40, 97)
    turb_removal  = np.clip(COD_removal + rng.uniform(5, 12, n),  45, 98)

    # ---- Nilai akhir setelah proses ----
    COD_final       = COD_init  * (1 - COD_removal   / 100)
    BOD_final       = BOD_init  * (1 - BOD_removal   / 100)
    TSS_final       = TSS_init  * (1 - TSS_removal   / 100)
    TDS_final       = TDS_init  * rng.uniform(0.55, 0.85, n)
    color_final     = color_init * (1 - color_removal / 100)
    turbidity_final = turbidity_init * (1 - turb_removal / 100)

    # pH cenderung bergerak mendekati netral setelah proses
    pH_shift = (pH_init - 7.0) * rng.uniform(0.2, 0.5, n)
    pH_final = np.clip(pH_init - pH_shift + rng.normal(0, 0.3, n), 5.5, 9.5)

    # Variabel tambahan untuk Isolation Forest
    power_consumption = voltage * current       # Watt
    inflow_rate       = rng.uniform(5, 50, n)  # L/menit

    # ---- Label bersih/tidak — Permen LHK No.16/2019 ----
    # Baku mutu: COD ≤100, BOD ≤50, TSS ≤200, TDS ≤2000, pH 6–9
    label_clean = (
        (COD_final <= 100) & (BOD_final <= 50) &
        (TSS_final <= 200) & (TDS_final <= 2000) &
        (pH_final  >= 6.0) & (pH_final  <= 9.0)
    ).astype(int)

    return pd.DataFrame({
        # Input: sensor kualitas awal
        'pH_init'              : np.round(pH_init, 2),
        'turbidity_init_NTU'   : np.round(turbidity_init, 1),
        'TDS_init_mgL'         : np.round(TDS_init, 1),
        'conductivity_uScm'    : np.round(conductivity, 1),
        'color_init_PtCo'      : np.round(color_init, 1),
        'COD_init_mgL'         : np.round(COD_init, 1),
        'BOD_init_mgL'         : np.round(BOD_init, 1),
        'temp_water_C'         : np.round(temp_water, 1),
        'TSS_init_mgL'         : np.round(TSS_init, 1),
        # Input: parameter proses
        'voltage_V'            : np.round(voltage, 1),
        'current_A'            : np.round(current, 2),
        'duration_min'         : np.round(duration).astype(int),
        'electrode_type'       : electrode_type,    # 0=Al, 1=Fe
        'electrode_gap_cm'     : np.round(electrode_gap, 1),
        'power_consumption_W'  : np.round(power_consumption, 1),
        'inflow_rate_Lmin'     : np.round(inflow_rate, 1),
        # Target: efisiensi penyisihan polutan
        'COD_removal_pct'      : np.round(COD_removal, 2),
        'BOD_removal_pct'      : np.round(BOD_removal, 2),
        'TSS_removal_pct'      : np.round(TSS_removal, 2),
        'color_removal_pct'    : np.round(color_removal, 2),
        'turbidity_removal_pct': np.round(turb_removal, 2),
        # Target: nilai kualitas air akhir
        'COD_final_mgL'        : np.round(COD_final, 1),
        'BOD_final_mgL'        : np.round(BOD_final, 1),
        'TSS_final_mgL'        : np.round(TSS_final, 1),
        'TDS_final_mgL'        : np.round(TDS_final, 1),
        'turbidity_final_NTU'  : np.round(turbidity_final, 1),
        'pH_final'             : np.round(pH_final, 2),
        # Target: label bersih/tidak
        'label_clean'          : label_clean,
    })


# ============================================================
# DATASET 1: WATER QUALITY
# ============================================================
# Batch A: distribusi umum — COD 300–21000 mg/L, parameter bervariasi
# Mewakili kondisi operasi harian yang beragam
df_A = generate_water_samples(
    n=800,
    cod_range=(300, 21000),
    current_range=(0.5, 10),
    duration_range=(15, 90),
    seed_offset=0
)

# Batch B: kondisi optimal — COD rendah, arus & durasi tinggi
# Mewakili industri kecil atau limbah yang sudah pre-treatment
# Menghasilkan lebih banyak sampel dengan label BERSIH untuk keseimbangan
df_B = generate_water_samples(
    n=200,
    cod_range=(300, 1500),
    current_range=(7, 10),
    duration_range=(60, 90),
    seed_offset=99
)

# Gabung dan acak urutan
df_water = (
    pd.concat([df_A, df_B], ignore_index=True)
    .sample(frac=1, random_state=0)
    .reset_index(drop=True)
)

# ============================================================
# DATASET 2: ENERGY MANAGEMENT (Hybrid Solar + Wind)
# ============================================================
# Sistem referensi dari riset Indonesia:
#   Panel surya : 100 Wp (Imp 5.5A, Vmp 18.2V)
#   Turbin VAWT : 300W rated di 13 m/s, cut-in 2 m/s
#   Irradiance  : hingga 1320 W/m² (tengah hari tropis)
#   Angin       : distribusi Weibull, umumnya <10 m/s

rng2 = np.random.RandomState(7)
N_E  = 1000

time_min    = rng2.randint(0, 1440, N_E)
hour        = time_min // 60

# Irradiance: sinusoidal 6 pagi–6 sore, dikali faktor awan acak
solar_angle    = np.maximum(0.0, np.sin(np.pi * (hour - 6) / 12))
cloud_cover    = rng2.uniform(0.3, 1.0, N_E)
irradiance     = np.where(
    (hour >= 6) & (hour <= 18),
    solar_angle * 1320 * cloud_cover,
    0.0
)

# Output panel 100 Wp (efisiensi 13–18%)
solar_output_W = np.clip(irradiance * rng2.uniform(0.13, 0.18, N_E), 0, 100)

# Angin: distribusi Weibull (shape=2, scale=3.5 → rata-rata ~3 m/s)
wind_speed     = np.clip(rng2.weibull(2.0, N_E) * 3.5, 0, 13)
wind_dir       = rng2.uniform(0, 360, N_E)

# Output VAWT 300W: hubungan kubik dengan kecepatan angin
wind_output_W  = np.where(
    wind_speed >= 2.0,
    np.clip((wind_speed / 13)**3 * 300 * rng2.uniform(0.7, 0.95, N_E), 0, 300),
    0.0
)

battery_SOC     = rng2.uniform(20, 100, N_E)
air_temp        = rng2.uniform(22, 35, N_E)
ec_power_demand = rng2.uniform(10, 160, N_E)

# Logika alokasi energi: prioritas solar → angin → baterai
solar_alloc   = np.minimum(solar_output_W, ec_power_demand)
rem1          = np.maximum(0, ec_power_demand - solar_alloc)
wind_alloc    = np.minimum(wind_output_W, rem1)
rem2          = np.maximum(0, rem1 - wind_alloc)
battery_alloc = np.minimum(battery_SOC / 100 * 50, rem2)   # maks 50 Wh dari baterai

total_alloc   = solar_alloc + wind_alloc + battery_alloc
total_alloc   = np.where(total_alloc < 1e-6, 1.0, total_alloc)

sf  = solar_alloc   / total_alloc * 100 + rng2.normal(0, 1.5, N_E)
wf  = wind_alloc    / total_alloc * 100 + rng2.normal(0, 1.5, N_E)
bf  = battery_alloc / total_alloc * 100 + rng2.normal(0, 1.5, N_E)
tot = np.where(sf + wf + bf < 1e-6, 1.0, sf + wf + bf)

solar_frac   = np.clip(sf / tot * 100, 0, 100)
wind_frac    = np.clip(wf / tot * 100, 0, 100)
battery_frac = np.clip(bf / tot * 100, 0, 100)

# Prediksi energi 1 jam ke depan (Wh) — estimasi konservatif 85%
energy_fcast  = (
    solar_output_W * rng2.uniform(0.7, 1.0, N_E)
  + wind_output_W  * rng2.uniform(0.6, 1.0, N_E)
) * 0.85

df_energy = pd.DataFrame({
    # Input
    'time_minutes'          : time_min,
    'hour'                  : hour,
    'irradiance_Wm2'        : np.round(irradiance, 1),
    'solar_output_W'        : np.round(solar_output_W, 2),
    'wind_speed_ms'         : np.round(wind_speed, 2),
    'wind_direction_deg'    : np.round(wind_dir, 1),
    'wind_output_W'         : np.round(wind_output_W, 2),
    'battery_SOC_pct'       : np.round(battery_SOC, 1),
    'air_temp_C'            : np.round(air_temp, 1),
    'ec_power_demand_W'     : np.round(ec_power_demand, 1),
    # Target
    'solar_fraction_pct'    : np.round(solar_frac, 2),
    'wind_fraction_pct'     : np.round(wind_frac, 2),
    'battery_fraction_pct'  : np.round(battery_frac, 2),
    'energy_forecast_1h_Wh' : np.round(energy_fcast, 2),
})

# ============================================================
# SIMPAN CSV
# ============================================================
df_water.to_csv('dataset_water_quality.csv', index=False)
df_energy.to_csv('dataset_energy_management.csv', index=False)

# ============================================================
# LAPORAN
# ============================================================
clean = df_water['label_clean'].sum()
total = len(df_water)

print("=" * 60)
print("DATASET 1: WATER QUALITY")
print("=" * 60)
print(f"  Sampel      : {total}  |  Kolom: {df_water.shape[1]}")
print(f"  Input       : 16 kolom")
print(f"  Target      : 12 kolom (5 removal %, 6 nilai akhir, 1 label)")
print(f"  Label BERSIH: {clean} ({clean/total*100:.1f}%)  — imbalance normal,")
print(f"               gunakan class_weight='balanced' di Random Forest")
print(f"  COD removal : {df_water['COD_removal_pct'].min():.0f}%–{df_water['COD_removal_pct'].max():.0f}%  avg {df_water['COD_removal_pct'].mean():.1f}%")
print(f"  TSS removal : avg {df_water['TSS_removal_pct'].mean():.1f}%")
print()
print("=" * 60)
print("DATASET 2: ENERGY MANAGEMENT")
print("=" * 60)
print(f"  Sampel  : {len(df_energy)}  |  Kolom: {df_energy.shape[1]}")
print(f"  Input   : 10 kolom")
print(f"  Target  : 4 kolom (alokasi solar/angin/baterai + forecast Wh)")
print(f"  Solar avg output : {solar_output_W.mean():.1f} W")
print(f"  Wind avg output  : {wind_output_W.mean():.1f} W")
print(f"  Alokasi rata-rata: solar {solar_frac.mean():.0f}% | angin {wind_frac.mean():.0f}% | baterai {battery_frac.mean():.0f}%")
print()
print("File tersimpan:")
print("  dataset_water_quality.csv")
print("  dataset_energy_management.csv")