import cv2
import numpy as np

cap=cv2.VideoCapture(2)
while 1:
    ret,frame=cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    cv2.imshow("ds",frame)
    if cv2.waitKey(1) & 0XFF==ord('q'):
        break
cv2.destroyAllWindows()
cap.release()


