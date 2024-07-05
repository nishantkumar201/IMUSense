#include <Arduino_LSM6DSOX.h>

float AccX, AccY, AccZ;
float GyroX, GyroY, GyroZ;
float accAngleX, accAngleY, gyroAngleX, gyroAngleY, gyroAngleZ;
float roll, pitch, yaw;
float AccErrorX, AccErrorY, GyroErrorX, GyroErrorY, GyroErrorZ;
float elapsedTime, currentTime, previousTime;
int c = 0;

void setup() {
  Serial.begin(192000);

  if (!IMU.begin()) {
    Serial.println("Failed to initialize IMU!");
    while (1);
  }

  // Call this function if you need to get the IMU error values for your module
  calculate_IMU_error();
  delay(20);
}


void loop() {
  // === Read accelerometer data === //
  if (IMU.accelerationAvailable()) {
    IMU.readAcceleration(AccX, AccY, AccZ);
    // Calculating Roll and Pitch from the accelerometer data
    accAngleX = (atan(AccY / sqrt(pow(AccX, 2) + pow(AccZ, 2))) * 180 / PI) +1.63; // AccErrorX ~(0.58) See the calculate_IMU_error() custom function for more details
    accAngleY = (atan(-1 * AccX / sqrt(pow(AccY, 2) + pow(AccZ, 2))) * 180 / PI) + 0.56; // AccErrorY ~(-1.58)
  }
  // === Read gyroscope data === //
  if (IMU.gyroscopeAvailable()) {
    previousTime = currentTime;  // Previous time is stored before the actual time read
    currentTime = millis();      // Current time actual time read
    elapsedTime = (currentTime - previousTime) / 1000;  // Divide by 1000 to get seconds
    IMU.readGyroscope(GyroX, GyroY, GyroZ);
    GyroX += 0.00;  // GyroErrorX ~(-0.56)
    GyroY += 0.00;     // GyroErrorY ~(2)
    GyroZ += 0.00; // GyroErrorZ ~ (-0.8)
    gyroAngleX += GyroX * elapsedTime;  // deg/s * s = deg
    gyroAngleY += GyroY * elapsedTime;
    yaw += GyroZ * elapsedTime;
    roll = 0.96 * gyroAngleX + 0.04 * accAngleX;
    pitch = 0.96 * gyroAngleY + 0.04 * accAngleY;
    gyroAngleX = roll;
    gyroAngleY = pitch;

    Serial.print(roll);
    Serial.print("/");
    Serial.print(pitch);
    Serial.print("/");
    Serial.println(yaw);
  }
  delay(100);
}


void calculate_IMU_error() {
  // Read accelerometer values 1000 times
  while (c < 1000) {
    if (IMU.accelerationAvailable()) {
      IMU.readAcceleration(AccX, AccY, AccZ);
      // Sum all readings
      AccErrorX += atan(AccY / sqrt(pow(AccX, 2) + pow(AccZ, 2))) * 180 / PI;
      AccErrorY += atan(-1 * AccX / sqrt(pow(AccY, 2) + pow(AccZ, 2))) * 180 / PI;
      c++;
    }
    delay(5);
  }
  AccErrorX /= 1000;
  AccErrorY /= 1000;
  c = 0;
  // Read gyro values 1000 times
  while (c < 1000) {
    if (IMU.gyroscopeAvailable()) {
      IMU.readGyroscope(GyroX, GyroY, GyroZ);
      // Sum all readings
      GyroErrorX += GyroX / 131.0;
      GyroErrorY += GyroY / 131.0;
      GyroErrorZ += GyroZ / 131.0;
      c++;
    }
    delay(5); // Add a short delay to prevent overloading the I2C bus
  }
  // Divide the sum by 1000 to get the error value
  GyroErrorX /= 1000;
  GyroErrorY /= 1000;
  GyroErrorZ /= 1000;
  // Print the error values on the Serial Monitor
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

