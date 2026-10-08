# config.py — Konfigurasi backend sistem penjernihan limbah batik

# ---- Firebase ----
FIREBASE_DATABASE_URL    = "https://YOUR_PROJECT.firebaseio.com"
FIREBASE_CREDENTIALS_PATH = "serviceAccountKey.json"   # download dari Firebase Console

# ---- Telegram ----
TELEGRAM_BOT_TOKEN  = "YOUR_BOT_TOKEN"          # dari @BotFather
TELEGRAM_CHAT_IDS   = ["YOUR_CHAT_ID"]          # list → support multi-penerima

# ---- Model ----
MODELS_DIR = "models"   # path ke folder hasil train_models.py

# ---- Threshold ----
# Classifier label_clean sangat imbalanced (1.2%), threshold diturunkan dari 0.5
CLEAN_PROB_THRESHOLD  = 0.15

# ---- Path RTDB ----
# Ditulis oleh IoT
PATH_SENSOR_WATER   = "/sensors/water/latest"
PATH_SENSOR_ENERGY  = "/sensors/energy/latest"

# Ditulis oleh backend (dibaca oleh website)
PATH_PRED_WATER     = "/predictions/water/latest"
PATH_PRED_ENERGY    = "/predictions/energy/latest"
PATH_PRED_ANOMALY   = "/predictions/anomaly/latest"

# History (untuk grafik tren di website)
PATH_HIST_WATER     = "/predictions/water/history"
PATH_HIST_ENERGY    = "/predictions/energy/history"

# ---- Physics constants (dipakai oleh model_runner untuk derivasi) ----
PANEL_EFFICIENCY  = 0.15    # 15% efisiensi panel surya 100Wp
PANEL_MAX_W       = 100.0   # Watt
TURBINE_RATED_W   = 300.0   # Watt (VAWT rated di 13 m/s)
TURBINE_RATED_MS  = 13.0    # m/s
TURBINE_CUT_IN_MS = 2.0     # m/s
TURBINE_EFF       = 0.82    # efisiensi rata-rata