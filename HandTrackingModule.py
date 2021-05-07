import cv2
import mediapipe as mp
import time


class HandDetection:

    # creating methods
    # __init__ method will be called automatically without explicitly mentioning in the

    def __init__(self, mode = False, maxHands = 2, DetectionConfidence = 0.5, TrackingConfidence = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.DetectionConfidence = DetectionConfidence
        self.TrackingConfidence = TrackingConfidence

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode,self.maxHands,
                                        self.DetectionConfidence,self.TrackingConfidence)
        self.mpDraw = mp.solutions.drawing_utils



    def findHands(self,img,draw = True):
        # Converting image from BGR to RGB
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.result = self.hands.process(imgRGB)

        # multiple hands

        if self.result.multi_hand_landmarks:
            for handlms in self.result.multi_hand_landmarks:

                # looping for multiple hand landmark and drawing the hand connections to webcam output
                if draw:
                    self.mpDraw.draw_landmarks(img,handlms,self.mpHands.HAND_CONNECTIONS)
        return img

    def detectPosition(self, img, handNo = 0, draw = True):
        # List for Landmark positions
        lmList = []
        if self.result.multi_hand_landmarks:
            myHand = self.result.multi_hand_landmarks[handNo]
            # enumerate is useful for obtaining an indexed list
            # using this to get ID number from 0 to 20 (ie.,21)
            # and to get corresponding x,y,z positions
            for ID_num, LandMark in enumerate(myHand.landmark):
                # print(ID_num,LandMark)

                # .shape returns a tuple of the number of rows, columns, and channels (if the image is color)
                # extracting size of output window
                h, w, c = img.shape
                # multiplying h,w with x,y positions of landmark to get in terms of pixel values
                # x, y are the ratio of the image, when multiplied with width and height it gives pixel value
                centreX, centreY = (int(LandMark.x*w), int(LandMark.y*h))
                # print(ID_num, centreX, centreY)
                lmList.append([ID_num, centreX, centreY])
                # indexing 1st landmark and creating a circle at that landmark
                if draw:
                    cv2.circle(img,(centreX,centreY),5,(255,0,0),cv2.FILLED)
        return lmList



# __name__ (A Special variable) in Python for module creation

def main():
    cap = cv2.VideoCapture(0)

    detector = HandDetection() # initializing object

    # Frame rate per Second
    previousTime = 0
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.detectPosition(img)
        if len(lmList) != 0:
            print(lmList[4])

        currentTime =  time.time()
        fps = 1/(currentTime - previousTime)
        previousTime = currentTime
        cv2.putText(img,f'FPS:{int(fps)}',(100,100),cv2.FONT_HERSHEY_PLAIN,3,(0,0,255),3)

        cv2.imshow("Hand Track",img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ =="__main__":
    main()