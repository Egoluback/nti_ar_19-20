import cv2
import math

class ShapeDetector:
    def __init__(self):
        pass
        
    def detect(self, c):
        shape = 'unidentified'
        p = cv2.arcLength(c, True)
        print('p=',p)
        a = cv2.approxPolyDP(c, 0.035 * p, True)
        print(len(a),a)
        if (len(a)==3):
            shape = '2'
        elif (len(a) == 4):
            (x,y,w,h) = cv2.boundingRect(a)
            ar = w/h
            shape = '0'
            if (abs(ar-1)>0.05):
                shape = 'r'
        else:
            shape = '1'
        return shape