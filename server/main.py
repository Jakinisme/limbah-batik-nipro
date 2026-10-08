"""
main.py — Entry point backend sistem penjernihan limbah batik.

Alur data:
  IoT → Firebase RTDB (/sensors/...)
      → listener callback
      → model inference (model_runner)
      → tulis prediksi ke RTDB (/predictions/...)
      → Telegram alert (jika kondisi tertentu)
      → Website baca dari /predictions/ secara realtime via JS SDK

Jalankan:
  python main.py

Dependensi:
  pip install firebase-admin scikit-learn pandas numpy requests
"""

import time
import logging
from datetime import datetime, timezone

import firebase_client as fb
import model_runner     as mr
import telegram_notifier as tg
from config import (
    PATH_SENSOR_WATER, PATH_SENSOR_ENERGY,
    PATH_PRED_WATER,   PATH_PRED_ENERGY, PATH_PRED_ANOMALY,
    PATH_HIST_WATER,   PATH_HIST_ENERGY,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---- State yang di-track antar update ----
_models           = None    # semua model dimuat sekali di startup
_prev_label_clean = None    # untuk deteksi perubahan status air


def _ts() -> str:
    return datetime.now(timezone.utc).isoformat()


# ============================================================
# CALLBACK: DATA WATER QUALITY
# ============================================================
def on_water(event):
    """
    Dipanggil Firebase listener setiap /sensors/water/latest berubah.
    Listener berjalan di background thread — jaga agar tidak blocking lama.
    """
    global _prev_label_clean

    if not event.data or not isinstance(event.data, dict):
        return

    payload = event.data
    log.info(
        f"[Water] COD={payload.get('COD_mgL')} mg/L  "
        f"I={payload.get('current_A')} A  "
        f"t={payload.get('duration_min')} min"
    )

    try:
        pred = mr.run_water_inference(payload, _models)
    except Exception as exc:
        log.error(f"[Water] Inference error: {exc}")
        return

    # ---- Tulis ke RTDB ----
    out = {**pred, "ts": _ts()}
    fb.write(PATH_PRED_WATER, out)
    fb.push(PATH_HIST_WATER, out)

    fb.write(PATH_PRED_ANOMALY, {
        "is_anomaly"   : pred["is_anomaly"],
        "anomaly_score": pred["anomaly_score"],
        "ts"           : _ts(),
    })

    log.info(
        f"[Water] → label={pred['label_clean']}  "
        f"prob={pred['prob_clean']:.3f}  "
        f"anomaly={pred['is_anomaly']}  "
        f"COD_removal={pred['COD_removal_pct']:.1f}%"
    )

    # ---- Telegram: anomali (setiap kejadian) ----
    if pred["is_anomaly"]:
        log.warning("[Water] ⚠ Anomali! Kirim alert Telegram...")
        tg.alert_anomali(payload, pred["anomaly_score"])

    # ---- Telegram: status air (hanya saat BERUBAH) ----
    if pred["label_clean"] != _prev_label_clean:
        tg.alert_status_air(pred["label_clean"], pred["prob_clean"], pred)
        _prev_label_clean = pred["label_clean"]


# ============================================================
# CALLBACK: DATA ENERGY
# ============================================================
def on_energy(event):
    """Dipanggil setiap /sensors/energy/latest berubah."""
    if not event.data or not isinstance(event.data, dict):
        return

    payload = event.data
    log.info(
        f"[Energy] irr={payload.get('irradiance_Wm2')} W/m²  "
        f"wind={payload.get('wind_speed_ms')} m/s  "
        f"SOC={payload.get('battery_SOC_pct')}%"
    )

    try:
        pred = mr.run_energy_inference(payload, _models)
    except Exception as exc:
        log.error(f"[Energy] Inference error: {exc}")
        return

    out = {**pred, "ts": _ts()}
    fb.write(PATH_PRED_ENERGY, out)
    fb.push(PATH_HIST_ENERGY, out)

    log.info(
        f"[Energy] → solar={pred['solar_fraction_pct']:.1f}%  "
        f"wind={pred['wind_fraction_pct']:.1f}%  "
        f"batt={pred['battery_fraction_pct']:.1f}%  "
        f"forecast={pred['energy_forecast_1h_Wh']:.1f} Wh"
    )

    # Alert energi dikirim setiap update (silent — tidak bunyi notif)
    tg.alert_energi(pred)


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    log.info("=" * 55)
    log.info("  Backend Sistem Penjernihan Limbah Batik — START")
    log.info("=" * 55)

    # 1. Sambung Firebase
    log.info("Menghubungkan ke Firebase RTDB...")
    fb.init()

    # 2. Load semua model sekali (cache di memori)
    log.info("Loading model AI...")
    _models = mr.load_models()
    log.info("Semua model siap.")

    # 3. Pasang listeners
    log.info(f"Listening: {PATH_SENSOR_WATER}")
    fb.listen(PATH_SENSOR_WATER, on_water)

    log.info(f"Listening: {PATH_SENSOR_ENERGY}")
    fb.listen(PATH_SENSOR_ENERGY, on_energy)

    log.info("Backend aktif. Ctrl+C untuk berhenti.\n")

    # 4. Keep-alive — listeners jalan di background thread
    try:
        while True:
            time.sleep(30)
    except KeyboardInterrupt:
        log.info("Backend dihentikan.")