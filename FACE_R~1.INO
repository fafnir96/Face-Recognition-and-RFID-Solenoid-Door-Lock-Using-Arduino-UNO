#include <SPI.h>
#include <MFRC522.h>

#define SS_PIN 10
#define RST_PIN 9
#define RELAY 3 //relay pin
#define ACCESS_DELAY 2000
#define DENIED_DELAY 1000
int ap;
MFRC522 mfrc522(SS_PIN, RST_PIN);

void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);
  SPI.begin();
  mfrc522.PCD_Init();
  pinMode(RELAY, OUTPUT);
  digitalWrite(RELAY, LOW);
  Serial.println("Put your Card or Face Recognition");
  delay(2000);
}

void loop() {
  // put your main code here, to run repeatedly:
  // Baca Pakai UID
  bacaWAJAH();
  bacaUID();
}

void bacaWAJAH() {
  ap = Serial.parseInt();
  if (Serial.available() > 0) {
    Serial.println(ap);
    if (ap < 100) {
      Serial.println("Authorized access");
      digitalWrite(RELAY, HIGH);
      delay(ACCESS_DELAY);
      digitalWrite(RELAY, LOW);
    }
  } else if (Serial.available() < 0) {
    Serial.println(ap);
    if (ap < 0) {
      Serial.println(" Access denied");
      delay(DENIED_DELAY);
    }
  }
}

void bacaUID() {
  if ( ! mfrc522.PICC_IsNewCardPresent()) 
  {
    return;
  }
  // Select one of the cards
  if ( ! mfrc522.PICC_ReadCardSerial()) 
  {
    return;
  }
  //Show UID on serial monitor
  Serial.print("UID tag :");
  String content= "";
  byte letter;
  for (byte i = 0; i < mfrc522.uid.size; i++) 
  {
     Serial.print(mfrc522.uid.uidByte[i] < 0x10 ? " 0" : " ");
     Serial.print(mfrc522.uid.uidByte[i], HEX);
     content.concat(String(mfrc522.uid.uidByte[i] < 0x10 ? " 0" : " "));
     content.concat(String(mfrc522.uid.uidByte[i], HEX));
  }
  Serial.println();
  Serial.print("Message : ");
  content.toUpperCase();
  if (content.substring(1) == "47 8C E6 2E" || content.substring(1) == "23 97 B9 1B") //change here the UID of the card/cards that you want to give access
  {
    Serial.println("Authorized access");
    Serial.println();
    delay(500);
    digitalWrite(RELAY, HIGH);
    delay(ACCESS_DELAY);
    digitalWrite(RELAY, LOW);
  }
 
 else   {
    Serial.println(" Access denied");
    delay(DENIED_DELAY);
  }
}
