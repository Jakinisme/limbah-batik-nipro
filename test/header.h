#pragma once
#include <Arduino.h>

class TdsSensor {
public:
    TdsSensor(uint8_t pin, float vref = 3.3f, int sampleCount = 30);

    void begin();
    void update();               

    float getTdsValue() const;
    float getVoltage() const;
    void setTemperature(float temp); 

private:
    const uint8_t  _pin;
    const float    _vref;
    const int      _sampleCount;

    int*   _analogBuffer;
    int*   _analogBufferTemp;
    int    _analogBufferIndex = 0;

    float  _averageVoltage = 0.0f;
    float  _tdsValue       = 0.0f;
    float  _temperature    = 25.0f;

    unsigned long _lastSampleTime = 0;
    unsigned long _lastPrintTime  = 0;

    int   getMedianNum(int* arr, int len);
    float computeTds(float voltage);
};