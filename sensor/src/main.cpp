#include <Arduino.h>
#include "ph_sensor.h"

// ─── CONFIG ──────────────────────────────────────────────────
static constexpr uint8_t  PH_PIN        = 34;
static constexpr uint32_t READ_INTERVAL = 1000;   // ms
static constexpr uint32_t BAUD_RATE     = 115200;

// ─── EMA CONFIG ──────────────────────────────────────────────
// alpha: 0.1 = lebih smooth/lambat | 0.3 = lebih responsif
static constexpr float EMA_ALPHA = 0.15f;

// Threshold voltage ADC ESP32 dianggap saturasi
static constexpr float ADC_SATURATE_V = 3.28f;

// ─── Instance ────────────────────────────────────────────────
PHSensor phSensor(PH_PIN, 12, 3.3f);

float capturedV4 = 0.0f;
float capturedV7 = 0.0f;
float emaPH      = -1.0f;   // -1 = belum diinit

// ─── Helpers ─────────────────────────────────────────────────
void printBanner() {
    Serial.println(F("\n╔══════════════════════════════════╗"));
    Serial.println(F("║   pH Sensor 4502C — PlatformIO   ║"));
    Serial.println(F("╠══════════════════════════════════╣"));
    Serial.println(F("║  [4] Rekam voltage pH 4.0        ║"));
    Serial.println(F("║  [7] Rekam voltage pH 7.0        ║"));
    Serial.println(F("║  [c] Terapkan kalibrasi baru     ║"));
    Serial.println(F("║  [i] Info kalibrasi aktif        ║"));
    Serial.println(F("╚══════════════════════════════════╝\n"));
}

void printCalibrationInfo() {
    Serial.printf("[CAL] Slope     : %.6f\n", phSensor.slope());
    Serial.printf("[CAL] Intercept : %.6f\n", phSensor.intercept());
}

// ─── Serial command handler ───────────────────────────────────
void handleSerialCommand() {
    if (!Serial.available()) return;
    char cmd = static_cast<char>(Serial.read());

    switch (cmd) {
        case '4':
            capturedV4 = phSensor.readVoltage();
            Serial.printf("[CAL] Voltage @ pH 4.0 = %.4f V\n", capturedV4);
            break;
        case '7':
            capturedV7 = phSensor.readVoltage();
            Serial.printf("[CAL] Voltage @ pH 7.0 = %.4f V\n", capturedV7);
            break;
        case 'c':
            if (capturedV4 < 1e-3f || capturedV7 < 1e-3f) {
                Serial.println(F("[CAL] Rekam V4 dan V7 dulu!"));
            } else {
                phSensor.calibrate(capturedV4, capturedV7);
                emaPH = -1.0f;   // reset EMA setelah kalibrasi baru
                Serial.println(F("[CAL] Kalibrasi diterapkan."));
                printCalibrationInfo();
            }
            break;
        case 'i':
            printCalibrationInfo();
            break;
        default:
            break;
    }
}

// ─── Setup ───────────────────────────────────────────────────
void setup() {
    Serial.begin(BAUD_RATE);
    delay(500);

    phSensor.begin();
    phSensor.setSampling(10, 20);
    phSensor.calibrate(3.03f, 2.51f);   // default 5V system — kalibrasi ulang!

    printBanner();
    printCalibrationInfo();
    Serial.println();
}

// ─── Loop ────────────────────────────────────────────────────
void loop() {
    handleSerialCommand();

    static uint32_t lastRead = 0;
    const uint32_t  now      = millis();

    if (now - lastRead >= READ_INTERVAL) {
        lastRead = now;

        PHReading r = phSensor.read();

        if (!r.isValid()) {
            Serial.printf("[pH] ??.??  |  %.3f V  |  raw=%.2f  <- kalibrasi meleset\n",
                          r.voltage, r.rawPH);
            return;
        }

        // ── EMA smoothing ──
        if (emaPH < 0.0f) emaPH = r.ph;   // init pertama kali
        emaPH = EMA_ALPHA * r.ph + (1.0f - EMA_ALPHA) * emaPH;

        // ── Deteksi ADC saturasi ──
        bool saturated = (r.voltage >= ADC_SATURATE_V);

        // ── Output ──
        Serial.printf("[pH] %.2f (ema:%.2f)  |  %.3f V  |  %s%s\n",
                      r.ph,
                      emaPH,
                      r.voltage,
                      r.status(),
                      saturated ? "  << ADC JENUH, pH asli lebih rendah" : "");
    }
}