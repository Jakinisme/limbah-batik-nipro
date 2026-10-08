#pragma once

#include <Arduino.h>

// ─── Konstanta default ────────────────────────────────────────
static constexpr uint8_t  PH_DEFAULT_SAMPLES      = 10;
static constexpr uint32_t PH_DEFAULT_SAMPLE_DELAY = 20;   // ms

// ─── Hasil pembacaan ─────────────────────────────────────────
struct PHReading {
    float voltage;
    float ph;       // sudah di-clamp 0–14
    float rawPH;    // hasil formula mentah (sebelum clamp) — pakai untuk debug

    const char* status() const {
        if (ph < 6.0f)  return "ASAM";
        if (ph > 8.5f)  return "BASA";
        return "NETRAL";
    }

    // isValid() false berarti kalibrasi meleset atau probe tidak tercelup
    bool isValid() const {
        return rawPH >= 0.0f && rawPH <= 14.0f;
    }
};

// ─── Class PHSensor ──────────────────────────────────────────
class PHSensor {
public:
    // Constructor
    // @param pin          - GPIO pin analog (ESP32: pakai pin ADC-capable)
    // @param adcBits      - resolusi ADC (12 untuk ESP32, 10 untuk Arduino)
    // @param vref         - tegangan referensi ADC (3.3V ESP32, 5.0V Arduino)
    explicit PHSensor(uint8_t pin,
                      uint8_t adcBits = 12,
                      float   vref    = 3.3f);

    // Inisialisasi — panggil di setup()
    void begin();

    // Kalibrasi 2-titik dari voltage yang sudah diukur
    // @param voltageAtPH4 - voltage saat probe di buffer pH 4.0
    // @param voltageAtPH7 - voltage saat probe di buffer pH 7.0
    void calibrate(float voltageAtPH4, float voltageAtPH7);

    // Baca voltage mentah (rata-rata N sampel)
    float readVoltage() const;

    // Baca pH (pakai kalibrasi yang aktif)
    float readPH() const;

    // Baca voltage + pH sekaligus
    PHReading read() const;

    // Getter kalibrasi aktif
    float slope()     const { return _slope; }
    float intercept() const { return _intercept; }

    // Ubah jumlah sampel & delay sampling
    void setSampling(uint8_t count, uint32_t delayMs);

private:
    uint8_t  _pin;
    float    _vref;
    float    _adcMax;

    float    _slope;
    float    _intercept;

    uint8_t  _sampleCount;
    uint32_t _sampleDelay;

    float _voltageToRaw(float voltage) const;
};