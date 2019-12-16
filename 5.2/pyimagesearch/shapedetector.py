import cv2
import math

class ShapeDetector:
    def __init__(self, accuracy):
        self.accuracy = accuracy
        
    def detect(self, c):
        shape = 'unidentified'
        p = cv2.arcLength(c, True)
        print('p=',p)
        a = cv2.approxPolyDP(c, self.accuracy * p, True)
        print(len(a),a)
        if (len(a)==3):
            shape = '3'
        elif (len(a) == 4):
            (x,y,w,h) = cv2.boundingRect(a)
            ar = w/h
            print(abs(ar - 1))
            shape = '1'
            if (abs(ar-1) < 0.05):
                shape = '1'
        else:
            shape = '2'
        return shape