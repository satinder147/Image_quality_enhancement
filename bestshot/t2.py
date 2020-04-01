import cv2
import numpy as np
cap=cv2.VideoCapture(2)
saliency=None
c=0
while True:
    #print("ds")
    frame = cap.read()[1]
    frame=cv2.resize(frame,(500,500))
    if c%100==0:
        saliency=cv2.saliency.MotionSaliencyBinWangApr2014_create()
        saliency.setImagesize(frame.shape[1],frame.shape[0])
        saliency.init()
    gray = cv2.cv    print(i)tColor(frame, cv2.COLOR_BGR2GRAY)
    (success, saliencyMap) = saliency.computeSaliency(gray)
    saliencyMap = (saliencyMap * 255).astype("uint8")
    cv2.imshow("Map", saliencyMap)
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
cv2.destroyAllWindows()
cap.release()


'''
	# grab the frame from the threaded video stream and resize it
	# to 500px (to speedup processing)
	
	#frame = imutils.resize(frame, width=500)
    
	# if our saliency object is None, we need to instantiate it
	if saliency is None:
		saliency = cv2.saliency.MotionSaliencyBinWangApr2014_create()
		saliency.setImagesize(frame.shape[1], frame.shape[0])
		saliency.init()

    
	
	
	# display the image to our screen
	
	
	
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
# do a bit of cleanup
'''