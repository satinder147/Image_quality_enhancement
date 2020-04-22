import cv2
import dlib
from scipy.spatial import distance as dist
import numpy as np

# head angle
# scale of the face
# more parameters
# face occlusion


class Scoring:

    def __init__(self):
        self.blur_threshold = 3000

    @staticmethod
    def d(p1, p2):
        x = (p1.x, p1.y)
        y = (p2.x, p2.y)
        return dist.euclidean(x, y)

    @staticmethod
    def eye_aspect(self, points):
        p1, p2, p3, p4, p5, p6 = points
        eye_aspect = (self.d(p6) + self.d(p5)) / (2.0 * self.d(p4))
        # print(str(eye_aspect))
        return eye_aspect

    @staticmethod
    def teeth_detection(image, points):
        p1, p2, p3, p4 = points
        roi = image[p2.y:p4.y, p1.x:p3.x]
        gray_img = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # lower_white = np.array([0, 0, 0], dtype=np.uint8)
        # upper_white = np.array([0, 0, 255], dtype=np.uint8)
        gray_img = cv2.Canny(gray_img, 100, 150)
        # gray=cv2.inRange(gray,lower_white,upper_white)#cv2.threshold(gray,200,255,cv2.THRESH_BINARY)
        # gray=cv2.erode(gray,None,iterations=2), try otsu thresholding
        # gray=cv2.dilate(gray,None,iterations=2)
        # cv2.imshow("teeth", gray)
        # cv2.waitKey(1)
        contours, _ = cv2.findContours(gray_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # cv2.drawContours(frame,con,-1,(0,255,0),2)
            return True
        
        return False

    @staticmethod
    def smile_aspect(self, image, points):
        p1, p2, p3, p4 = points
        # aspect = self.d(p1, p3)/self.d(p2, p4)  # Aspect not changing much
        slope = (p1.y-p4.y)/(p1.x-p4.x) # slope can be used to predict if the person opened his/her mouth
        # But opened mouth is not always good. For a good smile teeth can be detected
        cv2.line(image, (p1.x, p1.y), (p4.x, p4.y), (255, 0, 0), 2)
        cv2.line(image, (p1.x, p1.y), (p3.x, p3.y), (255, 0, 0), 2)
        cv2.line(image, (p2.x, p2.y), (p4.x, p4.y), (255, 0, 0), 2)
        # print("aspect: {} and slope {}".format(aspect,slope))
        teeth = self.teeth_detection(points)
        print("slope", slope)  # >.32 smile
        return image, slope, teeth
    
    def saliency(self):
        pass

    @staticmethod
    def blur(grey):
        """
        Calculates the variance of the result,
        after applying the laplacian kernal on
        the image.
        """
        return cv2.Laplacian(grey, cv2.CV_64F).var()

    def lighting(self):
        # considering lightning would not be a problem
        pass
   
    # histogram equalization
    def score(self, image, eye_points, smile_points):

        # add condition when the person is not detected
        score = 0
        left_eye = self.eye_aspect(eye_points[0])
        right_eye = self.eye_aspect(eye_points[1])
        print(left_eye, right_eye)
        eye_threshold = 0.25
        if left_eye <= eye_threshold and right_eye <= eye_threshold:
            score = 0
            image = cv2.putText(image, "eyes closed", (50, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            print("both eyes closed")
        elif left_eye >= eye_threshold and right_eye >= eye_threshold:
            score += 50
            image = cv2.putText(image, "eyes open", (50, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            print("both eyes open")
        elif left_eye >= eye_threshold:
            score += 100
            image = cv2.putText(image, "left eye open", (50, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            print("left eye open")
        elif right_eye >= eye_threshold:
            score += 100
            image = cv2.putText(image, "right eye open", (50, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            print("right eye open")
        _, slope, teeth = self.smile_aspect(image, smile_points)
        if slope > 0.32 and teeth:
            score += 100
            image = cv2.putText(image, "smiling and mouth open", (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            print("mouth open and smiling")
        elif slope > 0.32:
            score -= 100
            image = cv2.putText(image, "mouth open but not smiling", (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            print("mouth open but not smiling")
        
        # global blur score
        blur_score = self.blur()
        if blur_score < self.blur_threshold:
            score -= 100
        # check if all the faces are in focus


def bb(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right()-x
    h = rect.bottom()-y
    return x, y, w, h


if __name__ == '__main__':
    obj = Scoring()
    predictor = dlib.shape_predictor("land.dat")
    detector = dlib.get_frontal_face_detector()
    name = 'sad.jpg'
    cap = cv2.VideoCapture(2)
    while 1:
        # ret,frame=cap.read()
        frame = cv2.imread(name, 1)
        w, h, c = frame.shape
        aspect = w/h
        frame = cv2.resize(frame, (480, int(aspect*480)))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray, 1)
        for (i, faces) in enumerate(faces):
            landmarks = predictor(gray, faces)
            lp = []
            rp = []
            for j in range(6):
                lp.append(landmarks.part(36+j))
                rp.append(landmarks.part(42+j))

            smile_points = [landmarks.part(48), landmarks.part(51), landmarks.part(54), landmarks.part(57)]
            obj.score(frame, [lp, rp], smile_points)
        cv2.imshow("Ds", frame)
        cv2.imwrite("sad.jpg", frame)
        cv2.waitKey(0)
        break
