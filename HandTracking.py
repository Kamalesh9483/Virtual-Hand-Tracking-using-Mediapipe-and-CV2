import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

# using hands module
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

# Frame rate per Second
previousTime = 0




while True:
    success, img = cap.read()
    # Converting image from BGR to RGB
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    result = hands.process(imgRGB)

    # multiple hands
    # print(result.multi_hand_landmarks)

    if result.multi_hand_landmarks:
        for handlms in result.multi_hand_landmarks:
            # enumerate is useful for obtaining an indexed list
            # using this to get ID number from 0 to 20 (ie.,21)
            #   and to get corresponding x,y,z positions
            for ID_num, LandMark in enumerate(handlms.landmark):
                # print(ID_num,LandMark)

                # for getting information of one landmark...
                # .shape returns a tuple of the number of rows, columns, and channels (if the image is color)

                # extracting size of output window
                h, w, c = img.shape
                # multiplying h,w with x,y positions of landmark to get in terms of pixel values
                # x, y are the ratio of the image, when multiplied with width and height it gives pixel value
                centreX, centreY = (int(LandMark.x*w), int(LandMark.y*h))
                print(ID_num, centreX, centreY)

                # indexing 1st landmark and creating a circle at that landmark
                if ID_num == 0:
                    cv2.circle(img,(centreX,centreY),10,(255,0,0),cv2.FILLED)


            # looping for multiple hand landmark and drawing the hand connections to webcam output
            mpDraw.draw_landmarks(img,handlms,mpHands.HAND_CONNECTIONS)



    currentTime =  time.time()
    fps = 1/(currentTime - previousTime)
    previousTime = currentTime
    cv2.putText(img,f'FPS:{int(fps)}',(100,100),cv2.FONT_HERSHEY_PLAIN,3,(0,0,255),3)

    cv2.imshow("Hand Track",img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break