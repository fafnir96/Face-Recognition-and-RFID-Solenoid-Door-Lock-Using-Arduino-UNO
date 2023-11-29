import cv2
import time
import serial
import threading

ARDUINO_SERIAL = serial.Serial('com4',9600)
time.sleep(2)
print (ARDUINO_SERIAL.readline())
print ("You have new message from Arduino")

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
face_cascade = cv2.CascadeClassifier('Cascades\haarcascade_frontalface_default.xml') 
eye_cascade = cv2.CascadeClassifier('Cascades\haarcascade_eye.xml') 
smile_cascade = cv2.CascadeClassifier('Cascades\haarcascade_smile.xml') 

font = cv2.FONT_HERSHEY_SIMPLEX

id = 0

names = ['Unknown', 'Irfan', 'Mario', 'Ferdian']

# Initialize and start realtime video capture
video_capture = cv2.VideoCapture(0) 
video_capture.set(3, 640) # set video widht
video_capture.set(4, 480) # set video height

# Define min window size to be recognized as a face
minW = 0.1*video_capture.get(3)
minH = 0.1*video_capture.get(4)

# apakah sedang terhubung dengan arduino
is_comm = False
# siklus / proses jeda baca
cycles = 0
THRESHOLD_CYCLES = 20 #cycles

DELAY_TIME = 4

def call_arduino():
        global is_comm
        global cycles
        try :
                ARDUINO_SERIAL.write("1".encode())
                print ("STATE->OPEN")
                time.sleep(DELAY_TIME)
                ARDUINO_SERIAL.write("0".encode())
                print ("STATE->OFF")
                is_comm = False
                cycles = 0
        except Exception as e :
                print(e)

def detect(gray, frame): 
        global is_comm
        global cycles

        faces = face_cascade.detectMultiScale( 
                gray,
                scaleFactor = 1.2,
                minNeighbors = 5,
                minSize = (int(minW), int(minH)),
               ) 
        for (x,y,w,h) in faces:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]
                eyes = eye_cascade.detectMultiScale(roi_gray)
                
                if(len(faces) > 0):
                        # makin kecil nilai confidence, maka hasil makin akurat
                        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
                        if (confidence < 100): 
                                id = names[id]
                                confidence = "   {0}%".format(round(100 - confidence))
                                
                                cycles += 1
                                if(cycles >= THRESHOLD_CYCLES) :
                                        if(is_comm != True):
                                                is_comm = True
                                                comm = threading.Thread(target=call_arduino, args=())
                                                comm.start()
                        else:
                                id = "Unknown"
                                confidence = "   {0}%".format(round(100 - confidence))

                                if(is_comm != True):
                                        ARDUINO_SERIAL.write("0".encode())
                                        cycles = 0
                                        
                        cv2.putText(
                            frame,
                            str(id),
                            (x+5, y-5),
                            font,
                            1,
                            (255,255,255),
                            2
                            )

                        cv2.putText(
                            frame,
                            str(confidence),
                            (x+5, y+h-5),
                            font,
                            1,
                            (255,255,0),
                            1
                            )
                        
                else :
                        cycles = 0

                for (ex,ey,ew,eh) in eyes:
                            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

        if(len(faces) < 1):
                if(is_comm != True):
                        ARDUINO_SERIAL.write("0".encode())
                        cycles = 0

        return frame 

while True: 
        # Captures video_capture frame by frame 
        _, frame = video_capture.read()  
        
        # To capture image in monochrome                     
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)   
          
        # calls the detect() function     
        canvas = detect(gray, frame)    
        
        # Displays the result on camera feed                      
        cv2.imshow('Video', canvas)  
        
        # The control breaks once q key is pressed                         
        if cv2.waitKey(1) & 0xff == ord('q'):                
             break

# Release the capture once all the processing is done. 
video_capture.release()                                  
cv2.destroyAllWindows() 
