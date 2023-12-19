#include <WiFi.h>
#include <ESPAsyncWebSrv.h>
#include <ESP32Servo.h>
#include <HTTPClient.h>

char* ssid = "VUONG HAI LONG";
char* password = "0977644651";
const char* serverAddress = "192.168.1.54";
const int serverPort = 8081;

const int irSensorPin1 = 4; 
const int irSensorPin2 = 5;

int previousIrSensorValue1 = HIGH; 
int previousIrSensorValue2 = HIGH;

AsyncWebServer server(80);
Servo myServo;

bool servoOpen = false;
unsigned long servoResetStartTime = 0;
const int servoResetDuration = 3000; 

String license_plate = "";

void setup() {
  Serial.begin(115200);
  myServo.attach(2);

  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);
    Serial.println("Connecting to WiFi...");
  }

  Serial.println("Connected to WiFi");
  Serial.print("IP address: ");
  Serial.println(WiFi.localIP());

  server.on("/openservo", HTTP_GET, [](AsyncWebServerRequest *request) {
    if (!servoOpen) {
      myServo.write(90);
      servoOpen = true;
      servoResetStartTime = millis(); // Record the start time for resetting the servo
      request->send(200, "text/plain", "Servo opened!");
    } else {
      request->send(400); // Send a bad request response if the servo is already open
    }
  });

  server.on("/setplate", HTTP_GET, [](AsyncWebServerRequest *request) {
    if (request->hasParam("plate")) {
      license_plate = request->getParam("plate")->value();
      request->send(200, "text/plain", "License plate updated!");
    } else {
      request->send(400, "text/plain", "No plate parameter provided!");
    }
  });

  server.begin();
  pinMode(irSensorPin1, INPUT);
  pinMode(irSensorPin2, INPUT);
}

void sendRequest(String param) {
  HTTPClient http;
  String url = "http://" + String(serverAddress) + ":" + String(serverPort) + "/api/senddata/" + param;
  http.begin(url); 
  Serial.println(url);
  int httpResponseCode = http.GET(); 
  if (httpResponseCode > 0) {
    license_plate = "";
    Serial.print("HTTP Response code: ");
    Serial.println(httpResponseCode);
    Serial.print("Sent param: ");
    Serial.println(param);
  } else {
    Serial.println("Error sending request");
  }
  http.end();
}

void loop() {
  delay(500);
  if (servoOpen && (millis() - servoResetStartTime >= servoResetDuration)) {
    myServo.write(0); // Reset the servo to its initial position after the specified duration
    servoOpen = false; // Reset flag
  }
  int irSensorValue1 = digitalRead(irSensorPin1);
  int irSensorValue2 = digitalRead(irSensorPin2);

  if (irSensorValue1 != previousIrSensorValue1) {
    if (license_plate == "") {
      sendRequest("A");
    } else {
      sendRequest("A:" + license_plate);
    }
    previousIrSensorValue1 = irSensorValue1;
  }

  if (irSensorValue2 != previousIrSensorValue2) {
    if (license_plate == "") {
      sendRequest("B");
    } else {
      sendRequest("B:" + license_plate);
    }
    previousIrSensorValue2 = irSensorValue2;
  }
}
