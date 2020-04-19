import cv2
img=cv2.imread("1.jpg",1)
img=cv2.resize(img,(400,600))
rows,cols = img.shape[0:2]
M = cv2.getRotationMatrix2D((cols/2,rows/2),10,1)
dst = cv2.warpAffine(img,M,(cols,rows))

cv2.imshow("ds",dst)
cv2.waitKey(0)