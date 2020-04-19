import cv2
import os
import numpy as np
path='data/final/mask/'
files=os.listdir(path)
a=[]
for i in files:
    img=cv2.imread(path+i,-1)
    a.append(img)
a=np.array(a)
np.save("mask.npy",a)
