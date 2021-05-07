
import cv2
import mediapipe as mp
import time
import HandTrackingModule as htm

cap = cv2.VideoCapture(0)

detector = htm.HandDetection() # initializing object

# Frame rate per Second
previousTime = 0
while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.detectPosition(img)
    if len(lmList) != 0:
        print(lmList[4])  # indexing for values of landmark 4 ie. thumb tip

    currentTime =  time.time()
    fps = 1/(currentTime - previousTime)
    previousTime = currentTime
    cv2.putText(img,f'FPS:{int(fps)}',(100,100),cv2.FONT_HERSHEY_PLAIN,3,(0,0,255),3)

    cv2.imshow("Hand Track",img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

