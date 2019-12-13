from pyimagesearch.shapedetector import ShapeDetector
import argparse
import imutils
import cv2

ap = argparse.ArgumentParser()
ap.add_argument('-i1','--i1', required = True, help = 'path to the input image 1')
ap.add_argument('-i2','--i2', required = True, help = 'path to the input image 2')
ap.add_argument('-i3','--i3', required = True, help = 'path to the input image 3')
args = vars(ap.parse_args())

resultArr = []

IMAGES = [cv2.imread(args['i1']), cv2.imread(args['i2']), cv2.imread(args['i3'])]


for image in IMAGES:
    resized = image
    ratio = 1
    print('open image')

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    thresh = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)[1]

    print('image process')

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    print('find countours')

    sd = ShapeDetector()

    for c in cnts:
        M = cv2.moments(c)
        cX = int((M['m10'] / M['m00'])* ratio)
        cY = int((M['m01'] / M['m00'])* ratio)
        shape = sd.detect(c)
        
        c = c.astype('float')
        c *= ratio
        c = c.astype('int')

        resultArr.append([shape, cX // 40, cY // 40])

with open("result.txt", "a") as file:
    toAdd = "["
    for shape in resultArr:
        shapeString = "("
        for el in shape:
            shapeString += str(el) + ","
        toAdd += shapeString[0 : -1] + "),"
    toAdd = toAdd[0 : -1]
    toAdd += "]"
    file.write(toAdd)

cv2.waitKey(0)