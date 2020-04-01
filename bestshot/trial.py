import cv2
import dlib
from scipy.spatial import distance as dist
import numpy as np

class scoring:


    def __init__(self):
        self.blur_threshold=3000

    def d(self,p1,p2):
        x=(p1.x,p1.y)
        y=(p2.x,p2.y)
        return dist.euclidean(x,y)
    

    def eye_aspect(self,points):
        p1,p2,p3,p4,p5,p6=points
        eye_aspect=(self.d(p2,p6)+self.d(p3,p5))/(2.0*self.d(p1,p4))
        #print(str(eye_aspect))
        return eye_aspect

    def teeth_detection(self,frame,points):
        p1,p2,p3,p4=points
        roi=frame[p2.y:p4.y,p1.x:p3.x]
        gray=cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
        gray=cv2.Canny(gray,100,150)
        #thresh,gray=cv2.threshold(gray,125,255,cv2.THRESH_BINARY)
        #gray=cv2.erode(gray,None,iterations=2)                          #try otsu thresholding
        #gray=cv2.dilate(gray,None,iterations=2)
        cv2.imshow("ds",gray)
        con,hei=cv2.findContours(gray,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        if(con):
            cv2.drawContours(frame,con,-1,(0,255,0),2)
            return True
        
        return False

    def smile_aspect(self,frame,points):
        p1,p2,p3,p4=points
        aspect=self.d(p1,p3)/self.d(p2,p4) #Aspect not changing much
        slope=(p1.y-p4.y)/(p1.x-p4.x) #slope can be used to predict if the person opened his/her mouth
        #But opened mouth is not always good. For a good smile teeth can be detected
        #cv2.line(frame,(p1.x,p1.y),(p4.x,p4.y),(255,0,0),2)
        #cv2.line(frame,(p1.x,p1.y),(p3.x,p3.y),(255,0,0),2)
        #cv2.line(frame,(p2.x,p2.y),(p4.x,p4.y),(255,0,0),2)
        #print("aspect: {} and slope {}".format(aspect,slope))
        teeth=self.teeth_detection(frame,points)
        #print("slope",slope) >.32 smile
        return frame,slope,teeth
    
    def saliency(self):
        pass

    def blur(self,grey):
        
        '''
        Calculates the variance of the result, 
        after applying the laplacian kernal on 
        the image.
        '''
        return cv2.Laplacian(grey,cv2.CV_64F).var()
        
    
    def lighting(self):
        #considering lightning would not be a problem
        pass
   
       #histogram equilization
    def score(self,frame,eye_points,smile_points):


        #add condition when the person is not detected
        photo_score=0
        left_eye=self.eye_aspect(eye_points[0])
        right_eye=self.eye_aspect(eye_points[1])

        if(left_eye<=0.25 and right_eye <=0.25):
            score=0
        elif(left_eye>=0.25):
            score+=100
        elif(right_eye>=0.25):
            score+=100
        else:
            score+=50
        
        _,slope,teeth=self.smile_aspect(frame,smile_points)
        if(slope>0.32 and teeth):
            score+=100
        elif(slope>0.32):
            score-=100
        
        #global blur score
        blur_score=self.blur(frame)
        if(blur_score<self.blur_threshold):
            score=-100
        #check if all the faces are in focus
        


def bb(rect):
    x=rect.left()
    y=rect.top()
    w=rect.right()-x
    h=rect.bottom()-y
    return (x,y,w,h)


obj=scoring()
predictor=dlib.shape_predictor("land.dat")
cap=cv2.VideoCapture(0)
ret=True
detector=dlib.get_frontal_face_detector()
while ret:
    ret,frame=cap.read()
    #frame=cv2.imread("../IMG_20200311_151935.jpg")
    w,h,c=frame.shape
    aspect=w/h
    frame=cv2.resize(frame,(480,int(aspect*480)))
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    rects=detector(gray,1)
    for (i,rects) in enumerate(rects):
        landmarks=predictor(gray,rects)
        left_eye=gray[landmarks.part(37).y:landmarks.part(40).y,landmarks.part(36).x:landmarks.part(39).x]
        right_eye=gray[landmarks.part(43).y:landmarks.part(46).y,landmarks.part(43).x:landmarks.part(45).x]
        #eye(right_eye,"right")
        lp=[]
        rp=[]
        for i in range(6):
            lp.append(landmarks.part(36+i))
            rp.append(landmarks.part(42+i))
        #ear=(eye(lp)+eye(rp))/2.0
        #print(ear)
        #if(ear<0.25):
        #    cv2.putText(frame,"close",(10,10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,145,0),2)
        #else:
        #    cv2.putText(frame,"open",(10,10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,145,0),2)
        cv2.rectangle(frame,(landmarks.part(36).x,landmarks.part(37).y),(landmarks.part(39).x,landmarks.part(40).y),(0,0,255),2)
        cv2.rectangle(frame,(landmarks.part(42).x,landmarks.part(43).y),(landmarks.part(45).x,landmarks.part(46).y),(0,0,255),2)
        '''
        for i in range(68):
            
            
            cv2.circle(frame,(landmarks.part(i).x,landmarks.part(i).y),3,(0,0,255),-3)
            #if i%2!=0:
            #cv2.putText(frame,str(i),(landmarks.part(i).x,landmarks.part(i).y),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,145,0),2)
        '''
        smilepoints=[landmarks.part(48),landmarks.part(51),landmarks.part(54),landmarks.part(57)]
        frame=obj.smile_aspect(frame,smilepoints)
        #obj.pose(frame,landmarks)
        x,y,w,h=bb(rects)
        
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.imshow("frame",frame)
    if cv2.waitKey(1) & 0XFF==ord("q"):
        break
cap.release()
cv2.destroyAllWindows()