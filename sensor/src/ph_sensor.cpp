#include "ph_sensor.h"

// ─── Constructor ─────────────────────────────────────────────
PHSensor::PHSensor(uint8_t pin, uint8_t adcBits, float vref)
    : _pin(pin),
      _vref(vref),
      _adcMax(static_cast<float>((1 << adcBits) - 1)),
      _slope(0.0f),
      _intercept(0.0f),
      _sampleCount(PH_DEFAULT_SAMPLES),
      _sampleDelay(PH_DEFAULT_SAMPLE_DELAY)
{
    calibrate(3.03f, 2.51f);   // default: 4502C powered 5V
}

// ─── begin() ─────────────────────────────────────────────────
void PHSensor::begin() {
#ifdef ARDUINO_ARCH_ESP32
    analogReadResolution(static_cast<uint8_t>(__builtin_ctz(
        static_cast<uint32_t>(_adcMax + 1)
    )));
    analogSetAttenuation(ADC_11db);   // Range: 0–3.9V input
#endif
    pinMode(_pin, INPUT);
}

// ─── calibrate() ─────────────────────────────────────────────
void PHSensor::calibrate(float voltageAtPH4, float voltageAtPH7) {
    if (fabsf(voltageAtPH7 - voltageAtPH4) < 1e-6f) return;

    _slope     = (7.0f - 4.0f) / (voltageAtPH7 - voltageAtPH4);
    _intercept = 7.0f - (_slope * voltageAtPH7);
}

// ─── readVoltage() ───────────────────────────────────────────
float PHSensor::readVoltage() const {
    long sum = 0;
    for (uint8_t i = 0; i < _sampleCount; i++) {
        sum += analogRead(_pin);
        delay(_sampleDelay);
    }
    float avg = static_cast<float>(sum) / _sampleCount;
    return avg * (_vref / _adcMax);
}

// ─── readPH() ────────────────────────────────────────────────
float PHSensor::readPH() const {
    float ph = _slope * readVoltage() + _intercept;
    return constrain(ph, 0.0f, 14.0f);
}

// ─── read() ──────────────────────────────────────────────────
PHReading PHSensor::read() const {
    PHReading result;
    result.voltage = readVoltage();
    result.rawPH   = _slope * result.voltage + _intercept;
    result.ph      = constrain(result.rawPH, 0.0f, 14.0f);
    return result;
}

// ─── setSampling() ───────────────────────────────────────────
void PHSensor::setSampling(uint8_t count, uint32_t delayMs) {
    _sampleCount = (count > 0) ? count : 1;
    _sampleDelay = delayMs;
}