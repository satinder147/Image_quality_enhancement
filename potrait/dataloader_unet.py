import albumentations as albu
import os
import cv2
import random
import numpy as np
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import segmentation_models_pytorch as smp
from numpy.random import permutation
from albumentations import (
    HorizontalFlip,    
    Compose,
    GridDistortion, 
    RGBShift,
    HueSaturationValue,
    OneOf,
    ChannelShuffle,
    CLAHE,
    RandomContrast,
    RandomBrightnessContrast,    
    RandomGamma,
    Rotate,
    RandomSunFlare,
    RandomFog,
    GridDropout,
    ToSepia,
    ToGray,
    RandomSnow,
    RandomSnow

)

aug3=HorizontalFlip(p=1)
aug6=Rotate(p=1,limit=45)
aug7=GridDistortion(p=1)
aug10=RandomFog(p=1)
aug11=RGBShift(p=1)
aug12=HueSaturationValue(p=1)
aug13=ChannelShuffle(p=1)
aug14=CLAHE(p=1)
aug15=RandomContrast(p=1)
aug16=RandomGamma(p=1)
aug17=ToGray(p=1)
aug18=ToSepia(p=1)
aug20=RandomSnow(p=1)

aug=Compose([OneOf([aug3,aug6,aug7]),OneOf([aug11,aug12,aug13]),
            OneOf([aug14,aug15,aug16]),OneOf([aug17,aug18]),
            OneOf([aug10,aug20])])



def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):

    transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(transform)


class load(Dataset):


    def __init__(self,**kwargs):

        self.width=kwargs["width"]
        self.height=kwargs["height"]
        self.imgs=np.load("img.npy")
        self.masks=np.load("mask.npy")
        self.samples=[]
        perm=permutation(self.imgs.shape[0])
        self.imgs=self.imgs[perm]
        self.masks=self.masks[perm]  
        self.pre= get_preprocessing(smp.encoders.get_preprocessing_fn('mobilenet_v2', 'imagenet'))
        self.transforms_img=transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
        self.transforms_mask=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.0,),(0.5,))])  #transforms.Grayscale(num_output_channels=1),
#                                                transforms.Normalize((0.5,),(0.5,))])

    def __len__(self):

        return int(self.imgs.shape[0])

    def __getitem__(self,idx):
        
        i=self.imgs[idx]
        j=self.masks[idx]
        #img=cv2.imread(self.path1+i,1)
        #img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        #img=cv2.blur(img,(3,3))
        #mask=cv2.imread(self.path2+j,0)
        #mask=cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
        #mask=cv2.Canny(mask,100,150)
        #mask=cv2.dilate(mask,None,iterations=5)
        img=cv2.resize(i,(self.width,self.height))
        #img=np.float32(img)/255.0
        mask=cv2.resize(j,(self.width,self.height))
        augmented=aug(image=img,mask=mask)
        img=augmented['image']
        mask=augmented['mask']
        ll=self.pre(image=img)
        img=ll['image']  
        mask=self.transforms_mask(mask)
        return (img,mask)
    
    def plot(self,img,name):
        img=np.transpose(img.numpy(),(1,2,0))
        img=img*0.5+0.5
        img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        cv2.imshow(name,img)
        


if(__name__=="__main__"):

    obj=load(width=128,height=128)
    res=obj.__getitem__(0)
    obj.plot(res[0],"image")
    obj.plot(res[1],"mask")
    cv2.waitKey(0)
    
