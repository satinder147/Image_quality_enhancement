import cv2
names=['wink.jpg']
for i in names:
    img=cv2.imread(i,1)
    h,w,_=img.shape
    w=300
    aspect=h/w
    h=int(aspect*w)
    print(w,h)
    img=cv2.resize(img,(w,h))
    img=img[0:300,0:300]
    
    cv2.imwrite(i,img)