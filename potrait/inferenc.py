#from runner2 import network
import torch
import torchvision
from torchvision import transforms
import cv2
from PIL import Image
import numpy as np
import segmentation_models_pytorch as smp
import albumentations as albu
import matplotlib.pyplot as plt
def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')



def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)




device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net=smp.Unet('mobilenet_v2')

net.load_state_dict(torch.load("aug58.pth"))
net.to(device)
transform= get_preprocessing(smp.encoders.get_preprocessing_fn('mobilenet_v2', 'imagenet'))

#transform=torchvision.transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
img=cv2.imread("images/9.jpg")
img=cv2.resize(img,(224,448))
print("ds")
cv2.imshow("im",img)
cv2.waitKey(0)


pc=transform(image=img)
img=torch.tensor(pc['image']).unsqueeze(0)
#img=transform(image=img)
#img=img['image']
#img=np.expand_dims(img,axis=0)
print(img.shape)
img=img.to(device)
res=net(img).detach().cpu().numpy()
res=np.transpose(res[0],(1,2,0))
res=np.uint8(res*0.5)
cv2.imshow("eds",res)
cv2.waitKey(0)

print(res.max(),res.min())
print(res.shape)
