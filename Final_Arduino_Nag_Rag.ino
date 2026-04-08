#include <Servo.h>

Servo s1, s2, s3;

String data = "";

void setup() {
  Serial.begin(9600);

  s1.attach(9);
  s2.attach(10);
  s3.attach(11);

  // HOME POSITION (theta = 0,0,0)
  s1.write(90);
  s2.write(90);
  s3.write(90);

  delay(2000);
}

void loop() {

  while (Serial.available()) {
    char c = Serial.read();

    if (c == '\n') {
      processData(data);
      data = "";
    } else {
      data += c;
    }
  }
}

void processData(String input) {

  int t1, t2, t3;

  // Read incoming values
  sscanf(input.c_str(), "%d,%d,%d", &t1, &t2, &t3);

  // 🔹 Move to TARGET
  moveSmooth(s1, t1);
  moveSmooth(s2, t2);
  moveSmooth(s3, t3);

  // 🔹 Stay at target for 3 seconds
  delay(3000);

  // 🔹 RETURN TO HOME (IMPORTANT)
  moveSmooth(s1, 90);
  moveSmooth(s2, 90);
  moveSmooth(s3, 90);
}

void moveSmooth(Servo &servo, int target) {

  int current = servo.read();

  if (current < target) {
    for (int i = current; i <= target; i++) {
      servo.write(i);
      delay(100);
    }
  } else {
    for (int i = current; i >= target; i--) {
      servo.write(i);
      delay(100);
    }
  }
}