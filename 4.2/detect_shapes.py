from pyimagesearch.shapedetector import ShapeDetector
import argparse
import imutils
import cv2

# DETECT SHAPES

TRESHOLD = 160
GRID_PARAM = 40

# getting args from user
ap = argparse.ArgumentParser()
ap.add_argument('-logPath','--logPath', required = True, help = 'path to the log file')
ap.add_argument('-i1','--i1', required = True, help = 'path to the input image 1')
ap.add_argument('-i2','--i2', required = True, help = 'path to the input image 2')
ap.add_argument('-i3','--i3', required = True, help = 'path to the input image 3')
args = vars(ap.parse_args())

resultArr = []

LOG_PATH = args['logPath'] # path to log file

IMAGES = [cv2.imread(args['i1']), cv2.imread(args['i2']), cv2.imread(args['i3'])]


for image in IMAGES:
    print('open image')

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # converting image to grayscale

    thresh = cv2.threshold(gray, TRESHOLD, 255, cv2.THRESH_BINARY)[1] # binary threshold setting

    print('image process')

    # getting contours of shapes
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
    cnts = imutils.grab_contours(cnts)

    print('find countours')

    sd = ShapeDetector()

    for c in cnts:
        # getting centres of shapes
        M = cv2.moments(c)
        cX = int((M['m10'] / M['m00']))
        cY = int((M['m01'] / M['m00']))
        shape = sd.detect(c) # analyzing contours
        
        c = c.astype('float')
        c = c.astype('int')

        resultArr.append([shape, cX // GRID_PARAM, cY // GRID_PARAM])


# writing to log file
with open(LOG_PATH, "w+") as file:
    toAdd = "["

    for shape in resultArr:
        items = list(map(str, shape))
        toAdd+='(' + ','.join(items) + '),'

    toAdd += "]"
    file.write(toAdd)
    file.close()