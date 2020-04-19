import numpy as np
from PIL import Image
import keras
import sys
from keras.models import load_model
import matplotlib.pyplot as plt
import cv2
import os


class potrait:


    def __init__(self):
        self.model=load_model('potrait/models/deconv_bnoptimized_munet.h5')

    def smooth(self,e1,e2,mask):
        val=(mask-e1)/(e2-e1)
        x=np.clip(val,0.0,1.0)
        return x*x*(3-2*x)

    def hisEqulColor(self,img):
        ycrcb=cv2.cvtColor(img,cv2.COLOR_BGR2YCR_CB)
        channels=cv2.split(ycrcb)
        #print len(channels)
        cv2.equalizeHist(channels[0],channels[0])
        cv2.merge(channels,ycrcb)
        cv2.cvtColor(ycrcb,cv2.COLOR_YCR_CB2BGR,img)
        return img

    def func(self,person,mask,name):

        # #person=cv2.imread("imgs/"+name,1)
        h1,w1,c=person.shape
        # aspect=float(w1/h1)
        # h=400
        # w=int(aspect*h)
        # print(h,w)
        # person=cv2.resize(person,(w,h))
        # kernal=np.ones((5,5),np.uint8)
        B=np.full((h1,w1),48)
        G=np.full((h1,w1),191)
        R=np.full((h1,w1),251)
        bg=np.dstack((B,G,R))
        bg=np.uint8(bg)
        # lower=np.array([0,0,179])
        # upper=np.array([179,34,255])
        # hsv=cv2.cvtColor(person,cv2.COLOR_BGR2HSV)
        # mask=cv2.inRange(hsv,lower,upper)
        # mask_inv=cv2.bitwise_not(mask)
        # mask=mask_inv
        # mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernal)
        # mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernal)
        # mask=cv2.dilate(mask,None,iterations=2)
        # mask=cv2.erode(mask,None,iterations=2)
        # person=cv2.imread("imgs/"+name,1)
        # mask=cv2.resize(mask,(w1,h1))
        # print(mask.shape)
        # mask=mask.reshape((h1,w1,1))
        # mask=np.float32(mask)/255.0
        #cv2.imshow("pers",person)
        #cv2.imshow("per",bg)
        #cv2.waitKey(0)
        person=np.float32(person)/255.0
        bg=np.float32(bg)/255.0
        res=mask*person + (1-mask)*bg
        res=np.uint8(res*255)
        cv2.imwrite("results/over"+name+".jpg",res)
        #cv2.imshow("res",res)

        #cv2.imshow("mask",mask)
        #cv2.waitKey(0)
    

    def getPotrait(self,img,name):
        h,w,_=img.shape
        aspect=w/h
        h=500
        w=int(aspect*h)
        img=cv2.resize(img,(w,h))
        frame=img.copy()
        cv2.imwrite("results/original"+name+".jpg",frame)
        img=cv2.resize(img,(128,128))/255.0
        mask=self.model.predict(img.reshape(1,128,128,3))
        mask=np.reshape(mask,(128,128,1))
        mask=mask*255
        mask=mask.astype("uint8")
        t,mask=cv2.threshold(mask,128,255,cv2.THRESH_BINARY)
        mask=cv2.morphologyEx(mask, cv2.MORPH_OPEN, (5,5))
        mask=cv2.morphologyEx(mask, cv2.MORPH_CLOSE, (5,5))
        mask=cv2.resize(mask,(w,h))
        mask=cv2.GaussianBlur(mask,(7,7),1)
        mask_inv=cv2.bitwise_not(mask)
        #cv2.imshow("dds",mask)
        val=self.smooth(0.3,0.5,mask)
        #cv2.imshow("ds",val)
        print(w,h)
        blurred=frame.copy()
        blurred=cv2.GaussianBlur(blurred,(9,9),0)
        background=cv2.bitwise_and(blurred,blurred,mask=mask_inv)
        foreground=cv2.bitwise_and(frame,frame,mask=mask)
        val=val.reshape((h,w,1))
        print(val.max())
        self.func(frame,val,name)
        background=np.float32(background)/255.0
        foreground=np.float32(foreground)/255.0
        result=val*foreground+background*(1-val)
        result=np.uint8(result*255.0)
        val=np.uint8(val*255)
        #result=self.hisEqulColor(result)
        #output=cv2.seamlessClone(result, blurred, val, (int(w/2),int(h/2)), cv2.NORMAL_CLONE)
        cv2.imwrite("results/result"+name+".jpg",result)
        #cv2.imwrite("output.jpg",output)
        cv2.imwrite("results/mask"+name+".jpg",val)
        #cv2.waitKey(0)

        
if __name__=="__main__":
    path="images/"
    obj=potrait()
    fold=os.listdir(path)
    for i in fold:
        img=cv2.imread(path+i,1) 
        obj.getPotrait(img,i.split(".")[0])
        print("done")

