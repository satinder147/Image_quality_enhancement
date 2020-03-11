
'''import cv2
from unet import Unet
import numpy as np
import torch

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
img=cv2.imread("12.jpg",1)
img=cv2.resize(img,(256,256))
model=Unet(3,1)
model=model.to(device)
img=img.to(device)
img=np.transpose(img,(1,2,0))
img=img.unsqueeze(0)

print(img.shape)

'''
from keras.preprocessing.image import img_to_array
import numpy as np
import cv2
from unet_keras import Models

models=Models(256,256,3)
model=models.arch3()
model.load_weights("road3.MODEL")
frame=cv2.imread("IMG_20200126_183320.jpg",1)
frame=cv2.resize(frame,(256,256))
frame=img_to_array(frame)
frame=frame.astype('float')/255.0
frame=np.expand_dims(frame,axis=0)
img=model.predict(frame)[0]
img=img*255
img2=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
t,img2=cv2.threshold(img2,200,255,cv2.THRESH_BINARY)
img2=img2.astype("uint8")
cv2.imshow("unet output",img2)
cv2.waitKey(0)
