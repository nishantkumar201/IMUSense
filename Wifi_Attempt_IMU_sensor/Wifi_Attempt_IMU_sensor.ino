#include <SPI.h>
#include <WiFiNINA.h>
#include <Arduino_LSM6DSOX.h>

char ssid[] = "";     // replace with your Wi-Fi network SSID
char pass[] = "";    // replace with your Wi-Fi network password

WiFiServer server(80);            // Create a server instance listening on port 80

float AccX, AccY, AccZ;
float GyroX, GyroY, GyroZ;
float accAngleX, accAngleY, gyroAngleX, gyroAngleY, gyroAngleZ;
float roll, pitch, yaw;
float AccErrorX, AccErrorY, GyroErrorX, GyroErrorY, GyroErrorZ;
float elapsedTime, currentTime, previousTime;
int c = 0;

void setup() {
  Serial.begin(9600);
  while (!Serial);

  if (!IMU.begin()) {
    Serial.println("Failed to initialize IMU!");
    while (1);
  }

  Serial.println("Attempting to connect to Wi-Fi network...");
  while (WiFi.begin(ssid, pass) != WL_CONNECTED) {
    Serial.print(".");
    delay(500);
  }

  Serial.println("\nConnected to Wi-Fi");
  Serial.print("IP Address: ");
  Serial.println(WiFi.localIP());

  server.begin();      

  calculate_IMU_error(); 
  delay(20);
}

void loop() {
  WiFiClient client = server.available();

  if (client) {
    Serial.println("New client connected");

    String request = client.readStringUntil('\r');
    Serial.println("Request: " + request);
    client.println("HTTP/1.1 200 OK");
    client.println("Content-Type: text/html");
    client.println();
    client.println("<html><body>");
    client.println("<h1>IMU Data</h1>");
    client.print("<p>Roll: ");
    client.print(roll);
    client.print("</p><p>Pitch: ");
    client.print(pitch);
    client.print("</p><p>Yaw: ");
    client.print(yaw);
    client.println("</p>");
    client.println("</body></html>");
    delay(10);
    client.stop();
    Serial.println("Client disconnected");
  }

  if (IMU.accelerationAvailable()) {
    IMU.readAcceleration(AccX, AccY, AccZ);
    accAngleX = (atan(AccY / sqrt(pow(AccX, 2) + pow(AccZ, 2))) * 180 / PI) + 1.63;
    accAngleY = (atan(-1 * AccX / sqrt(pow(AccY, 2) + pow(AccZ, 2))) * 180 / PI) + 0.56;
  }

  if (IMU.gyroscopeAvailable()) {
    previousTime = currentTime;
    currentTime = millis();
    elapsedTime = (currentTime - previousTime) / 1000;
    IMU.readGyroscope(GyroX, GyroY, GyroZ);
    GyroX += 0.00;
    GyroY += 0.00;
    GyroZ += 0.00;
    gyroAngleX += GyroX * elapsedTime;
    gyroAngleY += GyroY * elapsedTime;
    yaw += GyroZ * elapsedTime;
    roll = 0.96 * gyroAngleX + 0.04 * accAngleX;
    pitch = 0.96 * gyroAngleY + 0.04 * accAngleY;
    gyroAngleX = roll;
    gyroAngleY = pitch;
  }

  delay(100);
}

void calculate_IMU_error() {
  while (c < 1000) {
    if (IMU.accelerationAvailable()) {
      IMU.readAcceleration(AccX, AccY, AccZ);
      AccErrorX += atan(AccY / sqrt(pow(AccX, 2) + pow(AccZ, 2))) * 180 / PI;
      AccErrorY += atan(-1 * AccX / sqrt(pow(AccY, 2) + pow(AccZ, 2))) * 180 / PI;
      c++;
    }
    delay(5);
  }
  AccErrorX /= 1000;
  AccErrorY /= 1000;
  c = 0;
  
  while (c < 1000) {
    if (IMU.gyroscopeAvailable()) {
      IMU.readGyroscope(GyroX, GyroY, GyroZ);
      GyroErrorX += GyroX / 131.0;
      GyroErrorY += GyroY / 131.0;
      GyroErrorZ += GyroZ / 131.0;
      c++;
    }
    delay(5);
  }

  GyroErrorX /= 1000;
  GyroErrorY /= 1000;
  GyroErrorZ /= 1000;

  Serial.print("AccErrorX: ");
  Serial.println(AccErrorX);
  Serial.print("AccErrorY: ");
  Serial.println(AccErrorY);
  Serial.print("GyroErrorX: ");
  Serial.println(GyroErrorX);
  Serial.print("GyroErrorY: ");
  Serial.println(GyroErrorY);
  Serial.print("GyroErrorZ: ");
  Serial.println(GyroErrorZ);
}
