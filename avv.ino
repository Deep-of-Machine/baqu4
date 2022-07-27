
#include <Servo.h>



Servo myservo;  // create servo object to control a servo







void setup() {

  myservo.attach(9);  // 서보모터를 쉴드의 8번에 연결한다.





  Serial.begin(9600);
  myservo.write(90);
}



void loop() {
+
  while (Serial.available() > 0) {

    long value = Serial.parseInt();

    myservo.write(value);
  /*
    switch (value) {

      case 1:

        //posH = posH - 5; 같은 표현이다 posH -= 5;

        myservo.write(0);

        break;

      case 2:

        myservo.write(96);

        break;

      case 3:

        myservo.write(90);

        break;

      case 4:

        myservo.write(180);

        break;

    }

*/



  }

}
