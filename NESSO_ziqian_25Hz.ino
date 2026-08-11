// ============================================================
// EG2A17 GROUP 5 - NESSO ZIQIAN FINAL 25 Hz BLE SKETCH
// Raw IMU only. Detection is performed in the Python BLE gateway.
// BLE payload: Xg,Yg,Zg,Xdeg,Ydeg,Zdeg
// ============================================================

#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <Arduino_Nesso_N1.h>
#include <Arduino_BMI270_BMM150.h>

#define BLE_NAME "ziqian"
#define SERVICE_UUID "bdc766fc-7eee-417f-bbe0-2e71a8a2bf70"
#define COMBINED_IMU_UUID "f509416c-3c4b-401e-a768-b25a9e621a91"

const unsigned long SAMPLE_INTERVAL_MS = 40;  // 25 Hz

BLECharacteristic combinedCharacteristic(
  COMBINED_IMU_UUID,
  BLECharacteristic::PROPERTY_NOTIFY
);

BLEAdvertising* advertising = nullptr;
bool deviceConnected = false;
unsigned long lastSampleMs = 0;

class ServerCallbacks : public BLEServerCallbacks {
  void onConnect(BLEServer* server) override {
    deviceConnected = true;
    Serial.println("BLE connected");
  }

  void onDisconnect(BLEServer* server) override {
    deviceConnected = false;
    Serial.println("BLE disconnected; advertising restarted");
    if (advertising != nullptr) {
      advertising->start();
    }
  }
};

void setup() {
  Serial.begin(115200);
  delay(200);

  if (!IMU.begin()) {
    Serial.println("ERROR: BMI270 IMU failed to initialise");
    while (true) {
      delay(1000);
    }
  }

  BLEDevice::init(BLE_NAME);
  BLEServer* server = BLEDevice::createServer();
  server->setCallbacks(new ServerCallbacks());

  BLEService* service = server->createService(SERVICE_UUID);
  service->addCharacteristic(&combinedCharacteristic);
  service->start();

  advertising = BLEDevice::getAdvertising();
  advertising->addServiceUUID(SERVICE_UUID);
  advertising->setScanResponse(true);
  advertising->start();

  Serial.println("============================================");
  Serial.println("NESSO Group 5 final live sensor ready");
  Serial.println("BLE name: ziqian");
  Serial.println("Sampling: 25 Hz");
  Serial.println("Payload: Xg,Yg,Zg,Xdeg,Ydeg,Zdeg");
  Serial.println("Detection: Python BLE gateway, not NESSO edge");
  Serial.println("============================================");
}

void loop() {
  const unsigned long now = millis();

  if (now - lastSampleMs < SAMPLE_INTERVAL_MS) {
    delay(1);
    return;
  }

  // Advance by the fixed interval to keep the source cadence at 25 Hz.
  lastSampleMs += SAMPLE_INTERVAL_MS;
  if (now - lastSampleMs > SAMPLE_INTERVAL_MS * 4) {
    // Recover cleanly after a long pause rather than trying to catch up.
    lastSampleMs = now;
  }

  float xg = 0.0f, yg = 0.0f, zg = 0.0f;
  float xdeg = 0.0f, ydeg = 0.0f, zdeg = 0.0f;

  if (IMU.accelerationAvailable()) {
    IMU.readAcceleration(xg, yg, zg);
  } else {
    return;
  }

  if (IMU.gyroscopeAvailable()) {
    IMU.readGyroscope(xdeg, ydeg, zdeg);
  } else {
    return;
  }

  if (!deviceConnected) {
    return;
  }

  char payload[128];
  snprintf(
    payload,
    sizeof(payload),
    "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f",
    xg, yg, zg, xdeg, ydeg, zdeg
  );

  combinedCharacteristic.setValue((uint8_t*)payload, strlen(payload));
  combinedCharacteristic.notify();
}
