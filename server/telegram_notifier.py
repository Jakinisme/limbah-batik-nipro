"""
telegram_notifier.py
=====================
Kirim alert ke Telegram menggunakan Bot API.
Menggunakan requests langsung — tidak perlu library python-telegram-bot.

pip install requests
"""

import requests
from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_IDS

_BASE = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"


def _send(chat_id: str, text: str, disable_notification: bool = False):
    """Kirim pesan ke satu chat ID."""
    try:
        resp = requests.post(
            f"{_BASE}/sendMessage",
            json={
                "chat_id"              : chat_id,
                "text"                 : text,
                "parse_mode"           : "HTML",
                "disable_notification" : disable_notification,
            },
            timeout=10,
        )
        resp.raise_for_status()
    except Exception as e:
        print(f"[Telegram] Gagal kirim ke {chat_id}: {e}")


def broadcast(text: str, silent: bool = False):
    """Kirim pesan ke semua chat ID yang terdaftar di config."""
    for cid in TELEGRAM_CHAT_IDS:
        _send(cid, text, disable_notification=silent)


# ============================================================
# Template pesan
# ============================================================

def alert_anomali(sensor: dict, score: float):
    broadcast(
        "⚠️ <b>ANOMALI TERDETEKSI</b>\n"
        f"Skor anomali : <code>{score:.4f}</code>\n"
        f"COD awal     : <code>{sensor.get('COD_mgL', '?')} mg/L</code>\n"
        f"Arus         : <code>{sensor.get('current_A', '?')} A</code>\n"
        f"Durasi       : <code>{sensor.get('duration_min', '?')} menit</code>\n"
        f"Gap elektroda: <code>{sensor.get('electrode_gap_cm', '?')} cm</code>\n\n"
        "🔍 Periksa kondisi elektroda dan kalibrasi sensor."
    )


def alert_status_air(label_clean: int, prob: float, pred: dict):
    if label_clean:
        header = "✅ <b>EFFLUENT MEMENUHI BAKU MUTU</b>"
        note   = "Permen LHK No.16/2019 terpenuhi."
    else:
        header = "❌ <b>EFFLUENT BELUM MEMENUHI BAKU MUTU</b>"
        note   = "Pertimbangkan perpanjangan durasi atau peningkatan arus."

    broadcast(
        f"{header}\n"
        f"P(BERSIH): <code>{prob*100:.1f}%</code>\n\n"
        "📊 <b>Efisiensi Penyisihan:</b>\n"
        f"  COD       : <code>{pred.get('COD_removal_pct', 0):.1f}%</code>\n"
        f"  BOD       : <code>{pred.get('BOD_removal_pct', 0):.1f}%</code>\n"
        f"  TSS       : <code>{pred.get('TSS_removal_pct', 0):.1f}%</code>\n"
        f"  Warna     : <code>{pred.get('color_removal_pct', 0):.1f}%</code>\n"
        f"  Kekeruhan : <code>{pred.get('turbidity_removal_pct', 0):.1f}%</code>\n\n"
        f"📝 {note}"
    )


def alert_energi(pred: dict):
    solar = pred.get('solar_fraction_pct', 0)
    wind  = pred.get('wind_fraction_pct', 0)
    batt  = pred.get('battery_fraction_pct', 0)
    fcast = pred.get('energy_forecast_1h_Wh', 0)

    dominant = max(
        [('☀️ Solar', solar), ('🌬️ Angin', wind), ('🔋 Baterai', batt)],
        key=lambda x: x[1],
    )
    broadcast(
        f"⚡ <b>STATUS ENERGI</b> — Dominan: {dominant[0]}\n"
        f"  ☀️ Solar   : <code>{solar:.1f}%</code>  ({pred.get('solar_output_W', 0):.1f} W)\n"
        f"  🌬️ Angin  : <code>{wind:.1f}%</code>  ({pred.get('wind_output_W', 0):.1f} W)\n"
        f"  🔋 Baterai : <code>{batt:.1f}%</code>\n"
        f"  📈 Forecast 1 jam: <code>{fcast:.1f} Wh</code>",
        silent=True,   # energi = info rutin, jangan bunyi notif
    )