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
    cv2.imshow('Gray', gray)
    cv2.waitKey(0)

    thresh = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)[1]
    cv2.imshow('thresh', thresh)
    cv2.waitKey(0)
    print('image process')

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    print('find countours')

    sd = ShapeDetector()
    resultArr = []

    for c in cnts:
        M = cv2.moments(c)
        cX = int((M['m10'] / M['m00'])* ratio)
        cY = int((M['m01'] / M['m00'])* ratio)
        shape = sd.detect(c)
        
        c = c.astype('float')
        c *= ratio
        c = c.astype('int')
        resultArr.append([shape, cX // 40, cY // 40])
        cv2.putText(image, '('+str(shape)+','+str(cX//40)+','+str(cY//40)+')', (cX+10,cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow('Image', image)

    for i in range(0,100):
        image = cv2.line(image, (0,i*40), (4000,i*40), (255,255,0), 1)
        image = cv2.line(image, (i*40,0), (i*40,4000), (255,255,0), 1)

print('write file')
with open("result.txt", "w+") as file:
    toAdd = "["
    for shape in resultArr:
        items=list(map(str,shape))
        toAdd+='('+','.join(items)+'),'
    toAdd += "]"
    file.write(toAdd)
    file.close()