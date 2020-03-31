import numpy as np
from PIL import Image
import keras
import sys
from keras.models import load_model
import matplotlib.pyplot as plt
import cv2


class potrait:


    def __init__(self):
        self.model=load_model('models/deconv_bnoptimized_munet.h5')

    def getPotrait(self,img):
        h,w,_=img.shape
        aspect=w/h
        h=500
        w=int(aspect*h)
        img=cv2.resize(img,(w,h))
        frame=img.copy()
        img=cv2.resize(img,(128,128))/255.0
        mask=self.model.predict(img.reshape(1,128,128,3))
        mask=np.reshape(mask,(128,128,1))
        mask=mask*255
        mask=mask.astype("uint8")
        t,mask=cv2.threshold(mask,128,255,cv2.THRESH_BINARY)
        mask=cv2.resize(mask,(w,h))

        mask=cv2.blur(mask,(5,5))
        mask=cv2.morphologyEx(mask, cv2.MORPH_OPEN, (5,5))
        mask=cv2.morphologyEx(mask, cv2.MORPH_CLOSE, (5,5))
        mask_inv=cv2.bitwise_not(mask)

        blurred=frame.copy()
        blurred=cv2.GaussianBlur(blurred,(5,5),0)
        background=cv2.bitwise_and(blurred,blurred,mask=mask_inv)
        foreground=cv2.bitwise_and(frame,frame,mask=mask)
        result=cv2.bitwise_or(background,foreground)
        cv2.imwrite("result.jpg",result)

        
if __name__=="__main__":
    img=cv2.imread("../images/sh.jpg")
    cv2.imshow("im",img)
    cv2.waitKey(0)
    obj=potrait()
    obj.getPotrait(img)

