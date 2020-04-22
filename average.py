import cv2
import os
import numpy as np


def adjust_gamma(image, gamma=1.0):
    invgamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invgamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


def return_average(images):
    for i in range(len(images)):
        images[i] = np.float32(images[i]) / 255.0
    print(images[0].max())
    first = np.zeros((h, w, 3))
    for i in range(len(images)):
        first = first + images[i]
    print(first.max())
    result = first / len(images)
    return result


def increase(result):
    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
    hsv = np.int16(hsv)
    hsv[:, :, 2] = hsv[:, :, 2] * 4
    hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
    hsv = np.uint8(hsv)
    im = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return im


if __name__ == "__main__":
    files = os.listdir("average")
    images = []
    c = 0
    for i in files:
        img = cv2.imread("average/" + i, 1)
        img = cv2.resize(img, (600, 400))
        h, w, _ = img.shape
        if c == 0:
            cv2.imwrite("original.jpg", img)
        # img=img[0:h,0:int(w/2)]
        c = c + 1
        images.append(img)
    align = cv2.createAlignMTB()
    align.process(images, images)

    res = return_average(images)
    res = np.uint8(res * 255)
    cv2.imwrite("res.jpg", res)
    # res=adjust_gamma()
    # cv2.imshow("res",res)
    # cv2.waitKey(0)
    #
    # im=increase(res)
    # img=np.uint8(im*255)
    # cv2.imshow("img",im)
    # cv2.waitKey(0)
