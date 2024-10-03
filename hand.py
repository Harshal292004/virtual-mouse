import cv2
import mediapipe as mp 
import time


class handDetector():
     # Con => Confidence
    def __init__(self,mode=False,maxHands=2,detectionCon=0.5,trackCon=0.5):
        self.mode=mode
        self.maxHands=maxHands
        self.detectionCon=detectionCon
        self.trackCon=trackCon
          
        self.mpHands=mp.solutions.hands
        self.hands=self.mpHands.Hands(self.mode,self.maxHands,int(self.detectionCon),int(self.trackCon))
        self.tipIDs=[4,8,12,16,20]
        self.mpDraw=mp.solutions.drawing_utils          
    def  findHands(self,img,draw=True):
        imgRGB= cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results= self.hands.process(imgRGB) #hand detection and processing in real time 
        #if land marks are detected then they will be displayed with mpDraw
        if draw:
            if self.results.multi_hand_landmarks:
                #handLms (hand land marks )
                for handLms in self.results.multi_hand_landmarks:
                    self.mpDraw.draw_landmarks(img,handLms,self.mpHands.HAND_CONNECTIONS)
    def findPosition(self,img,handNo=0,points_list=[],draw=True):
        
        lmList=[]

        if self.results.multi_hand_landmarks:
            myHand= self.results.multi_hand_landmarks[handNo]
            for id,lm in enumerate(myHand.landmark):

                h,w,c=img.shape

                cx,cy=int(lm.x*w),int(lm.y*h)

                lmList.append([id,cx,cy])
                
                if draw:
                    if id in points_list:
                        cv2.circle(img,(cx,cy),5,(255,0,0),3,cv2.FILLED) 
        
        return lmList
    
    def fingersUp(self,lmList):
        if len(lmList) !=0:  # Ensure we have at least 9 landmarks (0-8)
            fingers=[]

            right_hand=False

            if lmList[0][1]<lmList[9][1]:
                right_hand=True
            else:
                right_hand=False


            if right_hand:
                for id in range(0,5):
                    if id==0:
                        if lmList[self.tipIDs[id]][1]<lmList[self.tipIDs[id]-1][1]:
                            fingers.append(0)
                        else:
                            fingers.append(1)
                    else:
                        if lmList[self.tipIDs[id]][2]<lmList[self.tipIDs[id]-2][2]:
                            fingers.append(1)
                        else:
                            fingers.append(0)
            else:
                for id in range(0,5):
                    if id==0:
                        if lmList[self.tipIDs[id]][1]>lmList[self.tipIDs[id]-1][1]:
                            fingers.append(0)
                        else:
                            fingers.append(1)
                    else:
                        if lmList[self.tipIDs[id]][2]<lmList[self.tipIDs[id]-2][2]:
                            fingers.append(1)
                        else:
                            fingers.append(0) 
            return fingers
        else:
            return None

def main():
    #previ time
    pTime= 0
    #current time
    cTime= 0

    detector=handDetector()
    #vedio capture 
    cap= cv2.VideoCapture(0)
   
    while True:
        success,img=cap.read()
        detector.findHands(img)
        lmList=detector.findPosition(img)
        if len(lmList)!=0:
            print(lmList[4])
        #gets the frames per seconds to display 
        cTime=time.time()
        fps=1/(cTime-pTime)
        pTime=cTime


        #(10,70)
        #is the positon
        #img,text,positon_of_text,Font of the text,scale,color,thickness 
        cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
        
        cv2.imshow("Image",img)
        cv2.waitKey(1)

if __name__=="__main__":
    main()