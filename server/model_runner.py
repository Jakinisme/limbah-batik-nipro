"""
model_runner.py
================
Load semua model AI satu kali saat startup, lalu jalankan inference
setiap ada data baru dari Firebase.

Fungsi utama:
  load_models()            → dict berisi semua model & scaler
  run_water_inference()    → RF Regressor + RF Classifier + Isolation Forest
  run_energy_inference()   → Multi-output RF Regressor
"""

import pickle
import numpy as np
import pandas as pd
from config import (
    MODELS_DIR,
    CLEAN_PROB_THRESHOLD,
    PANEL_EFFICIENCY, PANEL_MAX_W,
    TURBINE_RATED_W, TURBINE_RATED_MS, TURBINE_CUT_IN_MS, TURBINE_EFF,
)

# ---- Kolom fitur (harus identik dengan urutan saat training) ----
_WATER_COLS = [
    'pH_init', 'turbidity_init_NTU', 'TDS_init_mgL', 'conductivity_uScm',
    'color_init_PtCo', 'COD_init_mgL', 'BOD_init_mgL', 'temp_water_C',
    'TSS_init_mgL', 'voltage_V', 'current_A', 'duration_min',
    'electrode_type', 'electrode_gap_cm', 'power_consumption_W',
    'inflow_rate_Lmin', 'Q_coulomb',
]
_ISO_COLS     = _WATER_COLS[:-1]   # tanpa Q_coulomb (16 kolom asli)
_ENERGY_COLS  = [
    'irradiance_Wm2', 'solar_output_W', 'wind_speed_ms', 'wind_output_W',
    'battery_SOC_pct', 'air_temp_C', 'ec_power_demand_W',
    'hour_sin', 'hour_cos', 'wind_sin', 'wind_cos',
]
_TARGET_REMOVAL = [
    'COD_removal_pct', 'BOD_removal_pct', 'TSS_removal_pct',
    'color_removal_pct', 'turbidity_removal_pct',
]
_TARGET_ENERGY = [
    'solar_fraction_pct', 'wind_fraction_pct',
    'battery_fraction_pct', 'energy_forecast_1h_Wh',
]


def load_models() -> dict:
    """
    Load semua model dan scaler dari disk.
    Dipanggil SEKALI saat startup — simpan hasilnya di variabel global main.py.
    """
    def _load(name):
        with open(f'{MODELS_DIR}/{name}', 'rb') as f:
            return pickle.load(f)

    models = {
        'rf_reg'      : _load('rf_regressor_removal.pkl'),
        'rf_clf'      : _load('rf_classifier_clean.pkl'),
        'iso'         : _load('isolation_forest.pkl'),
        'scaler_iso'  : _load('scaler_isolation.pkl'),
        'rf_energy'   : _load('rf_regressor_energy.pkl'),
        'scaler_e'    : _load('scaler_energy.pkl'),
    }
    return models


def _derive_solar_output(irradiance_Wm2: float) -> float:
    """Estimasi output panel 100Wp dari irradiance."""
    return float(np.clip(irradiance_Wm2 * PANEL_EFFICIENCY, 0, PANEL_MAX_W))


def _derive_wind_output(wind_speed_ms: float) -> float:
    """Estimasi output VAWT 300W dari kecepatan angin (hubungan kubik)."""
    if wind_speed_ms < TURBINE_CUT_IN_MS:
        return 0.0
    return float(np.clip(
        (wind_speed_ms / TURBINE_RATED_MS) ** 3 * TURBINE_RATED_W * TURBINE_EFF,
        0, TURBINE_RATED_W
    ))


def run_water_inference(payload: dict, models: dict) -> dict:
    """
    Jalankan 3 model untuk data kualitas air.

    Payload Firebase (/sensors/water/latest):
      pH              : float  — pH electrode
      turbidity_NTU   : float  — turbidity sensor (NTU)
      TDS_mgL         : float  — TDS sensor (mg/L)
      conductivity_uScm: float — conductivity probe (μS/cm)
      color_PtCo      : float  — optical color sensor (Pt-Co)
      COD_mgL         : float  — estimasi COD dari UV-Vis / korelasi turb
      BOD_mgL         : float  — estimasi: COD × 0.4 (dihitung di firmware)
      temp_C          : float  — DS18B20 (°C)
      TSS_mgL         : float  — estimasi dari turbidity
      voltage_V       : float  — tegangan elektroda (V)
      current_A       : float  — ACS712 current sensor (A)
      duration_min    : int    — timer proses (menit)
      electrode_type  : int    — 0=Aluminium, 1=Besi (konfigurasi)
      electrode_gap_cm: float  — jarak elektroda (konfigurasi)
      inflow_rate_Lmin: float  — flow meter (L/menit)

    Returns dict:
      COD_removal_pct, BOD_removal_pct, TSS_removal_pct,
      color_removal_pct, turbidity_removal_pct,
      label_clean (0/1), prob_clean (float),
      is_anomaly (bool), anomaly_score (float)
    """
    p = payload
    power_W   = p['voltage_V'] * p['current_A']
    Q_coulomb = p['current_A'] * float(p['duration_min']) * 60.0

    # DataFrame fitur (17 kolom termasuk Q_coulomb)
    df = pd.DataFrame([{
        'pH_init'             : p['pH'],
        'turbidity_init_NTU'  : p['turbidity_NTU'],
        'TDS_init_mgL'        : p['TDS_mgL'],
        'conductivity_uScm'   : p['conductivity_uScm'],
        'color_init_PtCo'     : p['color_PtCo'],
        'COD_init_mgL'        : p['COD_mgL'],
        'BOD_init_mgL'        : p['BOD_mgL'],
        'temp_water_C'        : p['temp_C'],
        'TSS_init_mgL'        : p['TSS_mgL'],
        'voltage_V'           : p['voltage_V'],
        'current_A'           : p['current_A'],
        'duration_min'        : p['duration_min'],
        'electrode_type'      : p['electrode_type'],
        'electrode_gap_cm'    : p['electrode_gap_cm'],
        'power_consumption_W' : power_W,
        'inflow_rate_Lmin'    : p['inflow_rate_Lmin'],
        'Q_coulomb'           : Q_coulomb,
    }])[_WATER_COLS]

    # Model 1: RF Regressor — removal %
    pred_removal = models['rf_reg'].predict(df)[0]
    removal      = dict(zip(_TARGET_REMOVAL, pred_removal.round(2).tolist()))

    # Model 2: RF Classifier — label_clean
    prob_clean  = float(models['rf_clf'].predict_proba(df)[0, 1])
    label_clean = int(prob_clean >= CLEAN_PROB_THRESHOLD)

    # Model 3: Isolation Forest — anomaly detection
    X_iso        = df[_ISO_COLS]
    X_iso_scaled = models['scaler_iso'].transform(X_iso)
    iso_label    = int(models['iso'].predict(X_iso_scaled)[0])
    iso_score    = float(models['iso'].decision_function(X_iso_scaled)[0])

    return {
        **removal,
        'label_clean'  : label_clean,
        'prob_clean'   : round(prob_clean, 4),
        'is_anomaly'   : iso_label == -1,
        'anomaly_score': round(iso_score, 4),
        # Nilai estimasi akhir effluent (untuk ditampilkan di website)
        'COD_final_mgL'      : round(p['COD_mgL']         * (1 - removal['COD_removal_pct']         / 100), 1),
        'BOD_final_mgL'      : round(p['BOD_mgL']         * (1 - removal['BOD_removal_pct']         / 100), 1),
        'TSS_final_mgL'      : round(p['TSS_mgL']         * (1 - removal['TSS_removal_pct']         / 100), 1),
        'turbidity_final_NTU': round(p['turbidity_NTU']   * (1 - removal['turbidity_removal_pct']   / 100), 1),
        'TDS_final_mgL'      : round(p['TDS_mgL']         * 0.70, 1),  # rata-rata 30% removal TDS
    }


def run_energy_inference(payload: dict, models: dict) -> dict:
    """
    Jalankan model manajemen energi.

    Payload Firebase (/sensors/energy/latest):
      ts              : int    — Unix timestamp ms
      irradiance_Wm2  : float  — pyranometer / estimasi dari panel (W/m²)
      wind_speed_ms   : float  — anemometer (m/s)
      wind_dir_deg    : float  — wind vane (0–360°)
      battery_SOC_pct : float  — BMS / voltage divider (%)
      air_temp_C      : float  — DHT22 (°C)
      ec_demand_W     : float  — V×I proses elektrokoagulasi (W)

    solar_output_W dan wind_output_W TIDAK perlu dikirim IoT —
    dihitung otomatis dari irradiance dan wind_speed.

    Returns dict:
      solar_fraction_pct, wind_fraction_pct, battery_fraction_pct,
      energy_forecast_1h_Wh, solar_output_W, wind_output_W
    """
    p = payload

    # Derivasi dari raw sensor (tidak perlu dikirim IoT)
    solar_out = _derive_solar_output(p['irradiance_Wm2'])
    wind_out  = _derive_wind_output(p['wind_speed_ms'])

    # Jam dari timestamp (untuk cyclical encoding)
    ts_ms = p.get('ts', 0)
    from datetime import datetime, timezone
    hour  = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).hour if ts_ms else 12

    df = pd.DataFrame([{
        'irradiance_Wm2'   : p['irradiance_Wm2'],
        'solar_output_W'   : solar_out,
        'wind_speed_ms'    : p['wind_speed_ms'],
        'wind_output_W'    : wind_out,
        'battery_SOC_pct'  : p['battery_SOC_pct'],
        'air_temp_C'       : p['air_temp_C'],
        'ec_power_demand_W': p['ec_demand_W'],
        'hour_sin'         : np.sin(2 * np.pi * hour / 24),
        'hour_cos'         : np.cos(2 * np.pi * hour / 24),
        'wind_sin'         : np.sin(2 * np.pi * p['wind_dir_deg'] / 360),
        'wind_cos'         : np.cos(2 * np.pi * p['wind_dir_deg'] / 360),
    }])[_ENERGY_COLS]

    X_scaled = models['scaler_e'].transform(df)
    pred     = models['rf_energy'].predict(X_scaled)[0]
    result   = dict(zip(_TARGET_ENERGY, pred.round(2).tolist()))

    result['solar_output_W'] = round(solar_out, 2)
    result['wind_output_W']  = round(wind_out, 2)
    return result