import cv2
import os
path="images/"
files=os.listdir(path)
dic={}
for i in files:
    fold=os.listdir(path+i)
    dic[i]=[]
    for j in fold:
        img=cv2.imread(path+i+'/'+j,0)
        img=cv2.resize(img,(400,400))
        dic[i].append(cv2.Laplacian(img,cv2.CV_64F).var())
    
for i in dic.keys():
    print(i,sum(dic[i])/len(dic[i]))
    