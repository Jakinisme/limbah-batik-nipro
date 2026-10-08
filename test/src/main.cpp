#include <Arduino.h>

#define TDS_PIN 34

const float VREF = 3.3;          // Tegangan ADC ESP32
const int ADC_RESOLUTION = 4095; // 12-bit
float temperature = 25.0;        // Suhu air (°C)

const int SCOUNT = 30;
int analogBuffer[SCOUNT];
int analogBufferTemp[SCOUNT];

int getMedianNum(int bArray[], int iFilterLen)
{
    int bTab[iFilterLen];

    for (int i = 0; i < iFilterLen; i++)
    {
        bTab[i] = bArray[i];
    }

    for (int j = 0; j < iFilterLen - 1; j++)
    {
        for (int i = 0; i < iFilterLen - j - 1; i++)
        {
            if (bTab[i] > bTab[i + 1])
            {
                int temp = bTab[i];
                bTab[i] = bTab[i + 1];
                bTab[i + 1] = temp;
            }
        }
    }

    if ((iFilterLen & 1) > 0)
    {
        return bTab[(iFilterLen - 1) / 2];
    }
    else
    {
        return (bTab[iFilterLen / 2] + bTab[iFilterLen / 2 - 1]) / 2;
    }
}

void setup()
{
    Serial.begin(115200);

    analogReadResolution(12);

    Serial.println("DFRobot TDS Sensor Test");
}

void loop()
{
    for (int i = 0; i < SCOUNT; i++)
    {
        analogBuffer[i] = analogRead(TDS_PIN);
        delay(10);
    }

    memcpy(analogBufferTemp, analogBuffer, sizeof(analogBuffer));

    int medianADC = getMedianNum(analogBufferTemp, SCOUNT);

    float voltage = medianADC * VREF / ADC_RESOLUTION;

    // Temperature Compensation
    float compensationCoefficient = 1.0 + 0.02 * (temperature - 25.0);
    float compensationVoltage = voltage / compensationCoefficient;

    // DFRobot Official Formula
    float tdsValue =
        (133.42 * compensationVoltage * compensationVoltage * compensationVoltage
        - 255.86 * compensationVoltage * compensationVoltage
        + 857.39 * compensationVoltage)
        * 0.5;

    Serial.println("========== TDS SENSOR ==========");

    Serial.print("ADC                : ");
    Serial.println(medianADC);

    Serial.print("Voltage            : ");
    Serial.print(voltage, 3);
    Serial.println(" V");

    Serial.print("Comp. Voltage      : ");
    Serial.print(compensationVoltage, 3);
    Serial.println(" V");

    Serial.print("Temperature        : ");
    Serial.print(temperature, 1);
    Serial.println(" C");

    Serial.print("TDS                : ");
    Serial.print(tdsValue, 0);
    Serial.println(" ppm");

    Serial.println("================================");
    Serial.println();

    delay(1000);
}