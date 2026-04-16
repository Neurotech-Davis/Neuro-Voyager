
const int STANDBY = 3; //ON OFF switch
const int RIGHT_DIRECTION = 7; // Right Direction
const int RIGHT_SPEED = 5;  // Right Speed
const int LEFT_DIRECTION = 8; // Left Direction
const int LEFT_SPEED = 6;  // Left Speed

const int turnSpeed = 200;
const int moveSpeed = 200;

void setup() {
  pinMode(STANDBY, OUTPUT);
  pinMode(RIGHT_DIRECTION, OUTPUT);
  pinMode(RIGHT_SPEED, OUTPUT);
  pinMode(LEFT_DIRECTION, OUTPUT);
  pinMode(LEFT_SPEED, OUTPUT);
  
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
    case 'w': moveForward(); break;
    case 'a': turnLeft();    break;
    case 'd': turnRight();   break;
    case 's': moveBackwards();     break;
    case 'e': stopCar();     break;
  }
}

void moveForward() {
  digitalWrite(STANDBY, HIGH);
  // Set Directions
  digitalWrite(RIGHT_DIRECTION, HIGH);
  digitalWrite(LEFT_DIRECTION, HIGH);
  // Set Speeds (0-255)
  analogWrite(RIGHT_SPEED, moveSpeed);
  analogWrite(LEFT_SPEED, moveSpeed);
}

void turnLeft() {
  digitalWrite(STANDBY, HIGH);
  digitalWrite(RIGHT_DIRECTION, HIGH); // Right forward
  digitalWrite(LEFT_DIRECTION, LOW);  // Left backward
  analogWrite(RIGHT_SPEED, turnSpeed);
  analogWrite(LEFT_SPEED, turnSpeed);
}

void turnRight() {
  digitalWrite(STANDBY, HIGH);
  digitalWrite(RIGHT_DIRECTION, LOW);  // Right backward
  digitalWrite(LEFT_DIRECTION, HIGH); // Left forward
  analogWrite(RIGHT_SPEED, turnSpeed);
  analogWrite(LEFT_SPEED, turnSpeed);
}
void moveBackwards() {
  digitalWrite(STANDBY, HIGH);
  // Set Directions
  digitalWrite(RIGHT_DIRECTION, LOW);
  digitalWrite(LEFT_DIRECTION, LOW);
  // Set Speeds (0-255)
  analogWrite(RIGHT_SPEED, moveSpeed);
  analogWrite(LEFT_SPEED, moveSpeed);
}
void stopCar() {
  digitalWrite(STANDBY, LOW); // Easiest way to kill all power
  analogWrite(RIGHT_SPEED, 0);
  analogWrite(LEFT_SPEED, 0);
}


