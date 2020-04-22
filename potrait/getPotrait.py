import numpy as np
import cv2
import os
from segmentation import Segmentation


class Portrait:

    def __init__(self):
        self.segmentor = Segmentation()

    @staticmethod
    def smooth(e1, e2, mask):
        val = (mask-e1)/(e2-e1)
        x = np.clip(val, 0.0, 1.0)
        return x*x*(3-2*x)

    @staticmethod
    def his_equlcolor(image):
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)
        channels = cv2.split(ycrcb)
        # print len(channels)
        cv2.equalizeHist(channels[0], channels[0])
        cv2.merge(channels, ycrcb)
        cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR, image)
        return image

    @staticmethod
    def overlay(person, mask):

        h1, w1, c = person.shape
        blue = np.full((h1, w1), 48)
        green = np.full((h1, w1), 191)
        red = np.full((h1, w1), 251)
        bg = np.dstack((blue, green, red))
        bg = np.uint8(bg)
        person = np.float32(person)/255.0
        bg = np.float32(bg)/255.0
        res = mask*person + (1-mask)*bg
        res = np.uint8(res*255)
        cv2.imshow("res", res)
        cv2.waitKey(0)

    def get_portrait(self, image):
        input_shape = (224, 448)
        h, w, _ = image.shape
        aspect = w/h
        h = 500
        w = int(aspect*h)
        image = cv2.resize(image, (w, h))
        frame = image.copy()
        image = cv2.resize(image, input_shape)/255.0
        mask = self.segmentor.get_mask(img)
        mask = np.reshape(mask, (448, 224, 1))
        mask = mask*255
        mask = mask.astype("uint8")
        t, mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, (5, 5))

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        maxidx=0
        if contours is not None:
            areas = [cv2.contourArea(cnt) for cnt in contours]
            maxidx = np.argmax(areas)
        new_mask = np.zeros((448, 224), np.uint8)
        cv2.drawContours(new_mask, contours, maxidx, 255, 1)
        cv2.fillPoly(new_mask, pts=[contours[maxidx]], color=(255, 255, 255))
        # cv2.imshow("dsdsds", new_mask)
        mask = new_mask
        mask=cv2.erode(mask, None, iterations=1)
        mask = cv2.resize(mask, (w, h))
        mask = cv2.GaussianBlur(mask, (7, 7), 1)
        # cv2.imshow("original", img)
        mask_inv = cv2.bitwise_not(mask)
        # cv2.imshow("dds", mask)
        val = self.smooth(0.5, mask)
        # cv2.imshow("smoothed", val)
        # print(w, h)
        blurred = frame.copy()
        blurred = cv2.GaussianBlur(blurred, (9, 9), 0)
        background = cv2.bitwise_and(blurred, blurred, mask=mask_inv)
        foreground = cv2.bitwise_and(frame, frame, mask=mask)
        val = val.reshape((h, w, 1))
        background = np.float32(background)/255.0
        foreground = np.float32(foreground)/255.0
        result = val*foreground+background*(1-val)
        cv2.imshow("result", result)
        self.overlay(person=frame, mask=val)
        cv2.waitKey(0)

        
if __name__ == "__main__":
    path = "images/"
    obj = Portrait()
    img = cv2.imread("../images/potrait/3.jpg")
    obj.get_portrait(img)


