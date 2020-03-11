import cv2
import dlib
from scipy.spatial import distance as dist


class scoring:
    def __init__(self):
        pass

    def eye_aspect(self):
        pass

    def d(self,p1,p2):
        x=(p1.x,p1.y)
        y=(p2.x,p2.y)
        return dist.euclidean(x,y)
    

    def eye(self,points):
        p1,p2,p3,p4,p5,p6=points
        eye_aspect=(d(p2,p6)+d(p3,p5))/(2.0*d(p1,p4))
        #print(str(eye_aspect))
        return eye_aspect
    
   
        
       
    

def bb(rect):
    x=rect.left()
    y=rect.top()
    w=rect.right()-x
    h=rect.bottom()-y
    return (x,y,w,h)

predictor=dlib.shape_predictor("land.dat")
cap=cv2.VideoCapture(2)
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
        
        for i in range(68):
            cv2.circle(frame,(landmarks.part(i).x,landmarks.part(i).y),3,(0,0,255),-3)
            cv2.putText(frame,str(i),(landmarks.part(i).x,landmarks.part(i).y),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,145,0),2)
        
        x,y,w,h=bb(rects)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.imshow("frame",frame)
    if cv2.waitKey(1) & 0XFF==ord("q"):
        break
cap.release()
cv2.destroyAllWindows()