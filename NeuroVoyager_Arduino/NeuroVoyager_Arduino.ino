// ELEGOO Smart Robot Car V4.0 — TB6612FNG motor driver
// Pin definitions from: https://forum.arduino.cc/t/elegoo-robot-car-4-simple-code-examples-available/1106841
const int STBY = 3;    // Standby: LOW = off, HIGH = run
const int AIN = 7;     // Right motor direction: HIGH = forward, LOW = reverse
const int PWMA = 5;    // Right motor speed (PWM)
const int BIN1 = 8;     // Left motor direction: HIGH = forward, LOW = reverse
const int PWMB = 6;    // Left motor speed (PWM)

const int turnSpeed = 60;
const int moveSpeed = 60;

void setup() {
  pinMode(STBY, OUTPUT);
  pinMode(AIN, OUTPUT);
  pinMode(PWMA, OUTPUT);
  pinMode(BIN1, OUTPUT);
  pinMode(PWMB, OUTPUT);

  digitalWrite(STBY, HIGH); // Enable motors on startup

  Serial.begin(9600);
}

void loop() {
  if (Serial.available() > 0) {
    char command = Serial.read();

    Serial.print("I received: ");
    Serial.println(command);

    startMovement(command);
  }
}

void startMovement(char cmd) {
  switch (cmd) {
    case 'w': moveForward();   break;
    case 'a': turnLeft();      break;
    case 'd': turnRight();     break;
    case 's': moveBackwards(); break;
    case 'e': stopCar();       break;
  }
}

void moveForward() {
  digitalWrite(STBY, HIGH);
  digitalWrite(AIN, HIGH);   // Right forward
  digitalWrite(BIN1, HIGH);   // Left forward
  analogWrite(PWMA, moveSpeed);
  analogWrite(PWMB, moveSpeed);
}

void moveBackwards() {
  digitalWrite(STBY, HIGH);
  digitalWrite(AIN, LOW);    // Right reverse
  digitalWrite(BIN1, LOW);    // Left reverse
  analogWrite(PWMA, moveSpeed);
  analogWrite(PWMB, moveSpeed);
}

void turnLeft() {
  digitalWrite(STBY, HIGH);
  digitalWrite(AIN, HIGH);   // Right forward
  digitalWrite(BIN1, LOW);    // Left reverse
  analogWrite(PWMA, turnSpeed);
  analogWrite(PWMB, turnSpeed);
}

void turnRight() {
  digitalWrite(STBY, HIGH);
  digitalWrite(AIN, LOW);    // Right reverse
  digitalWrite(BIN1, HIGH);   // Left forward
  analogWrite(PWMA, turnSpeed);
  analogWrite(PWMB, turnSpeed);
}

void stopCar() {
  digitalWrite(STBY, LOW);
  analogWrite(PWMA, 0);
  analogWrite(PWMB, 0);
}
