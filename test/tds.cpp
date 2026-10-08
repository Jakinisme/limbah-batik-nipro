#include "header.h"
#include <string.h>  

TdsSensor::TdsSensor(uint8_t pin, float vref, int sampleCount)
    : _pin(pin), _vref(vref), _sampleCount(sampleCount)
{
    _analogBuffer     = new int[_sampleCount]();
    _analogBufferTemp = new int[_sampleCount]();
}

void TdsSensor::begin() {
    pinMode(_pin, INPUT);
}

void TdsSensor::setTemperature(float temp) {
    _temperature = temp;
}

float TdsSensor::getTdsValue() const { return _tdsValue; }
float TdsSensor::getVoltage()  const { return _averageVoltage; }

void TdsSensor::update() {
    unsigned long now = millis();

    if (now - _lastSampleTime >= 40UL) {
        _lastSampleTime = now;
        _analogBuffer[_analogBufferIndex++] = analogRead(_pin);
        if (_analogBufferIndex >= _sampleCount) {
            _analogBufferIndex = 0;
        }
    }

    if (now - _lastPrintTime >= 800UL) {
        _lastPrintTime = now;

        memcpy(_analogBufferTemp, _analogBuffer,
               _sampleCount * sizeof(int));

        _averageVoltage = getMedianNum(_analogBufferTemp, _sampleCount)
                          * _vref / 4096.0f;

        _tdsValue = computeTds(_averageVoltage);

        Serial.print("TDS Value: ");
        Serial.print(_tdsValue, 0);
        Serial.println(" ppm");
    }
}


float TdsSensor::computeTds(float voltage) {
    float coeff = 1.0f + 0.02f * (_temperature - 25.0f);
    float v = voltage / coeff;

    return (133.42f * v * v * v
           - 255.86f * v * v
           + 857.39f * v) * 0.5f;
}

int TdsSensor::getMedianNum(int* arr, int len) {

    for (int j = 0; j < len - 1; j++) {
        for (int i = 0; i < len - j - 1; i++) {
            if (arr[i] > arr[i + 1]) {
                int tmp   = arr[i];
                arr[i]    = arr[i + 1];
                arr[i + 1] = tmp;
            }
        }
    }
    return (len & 1) ? arr[(len - 1) / 2]
                     : (arr[len / 2] + arr[len / 2 - 1]) / 2;
}