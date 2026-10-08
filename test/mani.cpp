#include <Arduino.h>
#include "header.h"

static constexpr uint8_t TDS_PIN  = 34;
static constexpr float   VREF     = 3.3f;
static constexpr int     SCOUNT   = 30;

TdsSensor tdsSensor(TDS_PIN, VREF, SCOUNT);

void setup() {
    Serial.begin(115200);
    tdsSensor.begin();
}

void loop() {
    tdsSensor.update();
}