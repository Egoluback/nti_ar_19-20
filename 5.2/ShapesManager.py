import numpy as np
import argparse
import imutils
import cv2
import copy

from accessify import private

from pyimagesearch.shapedetector import ShapeDetector

OBJECT_PARAMS = []

with open("log.txt", "r") as file:
	lines = file.readlines()
	print(lines)
	for line in lines:
		params = line.split(":")

		try:
			pos = list(map(int, params[0][params[0].find("(") + 1 : params[0].find(")")].split(",")))
			
			value = int(params[1])
		except: continue

		OBJECT_PARAMS.append([pos, value])

# print(OBJECT_PARAMS)

THRESHOLD = 200

class ShapesManager:
    def __init__(self, filePath):
        self.filePath = filePath

        self.field = np.load(self.filePath)
    
    @private
    def getPiece(self, inputField:np.array, r:int, c:int):
        field = copy.deepcopy(inputField[r * 200:(r + 1) * 200, c * 200:(c + 1) * 200])

        return field

    @private
    def getAverageBright(self, piece: np.array):
        return piece.sum() / piece.size

    @private
    def getThreshold(self, piece: np.array):
        averageBright = self.getAverageBright(piece)

        if (averageBright > 0 and averageBright <= 50):
            return 70
        elif (averageBright > 50 and averageBright <= 87):
            return 100
        elif (averageBright > 87 and averageBright <= 150):
            return 150
        elif (averageBright > 150 and averageBright <= 200):
            return 200
        elif (averageBright > 200 and averageBright <= 300):
            return 250

    def GetShapes(self):
        shapes = []
        
        cv2.imshow("Image", self.field / np.amax(self.field))
        
        maxValue = np.amax(self.field)
        
        self.field /= np.amax(maxValue)

        gridField = self.field
        
        for i in range(0, 100):
            gridField = cv2.line(gridField, (0,i*200), (2000,i*200), (255,255,0), 1)
            gridField = cv2.line(gridField, (i*200,0), (i*200,2000), (255,255,0), 1)
        
        cv2.imshow("Image", gridField)
        cv2.waitKey(0)

        for i in range(10):
            for j in range(10):
                print(i, j)
                piece = self.getPiece(self.field, i, j)

                piece *= 255
                piece = piece.astype(np.uint8)
                
                threshold = self.getThreshold(piece)

                print(threshold, self.getAverageBright(piece))

                # for obj in OBJECT_PARAMS:
                # 	if (obj[0] == [i, j]):
                # 		threshold = obj[1]
                
                thresh = cv2.threshold(piece, threshold, 255, cv2.THRESH_BINARY)[1]

                cv2.imshow("Image", thresh)
                cv2.waitKey(0)

                cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cnts = imutils.grab_contours(cnts)

                shapeDetector = ShapeDetector(0.16)

                result = 0

                if (len(cnts) > 0):
                    print("Object spotted")

                    result = shapeDetector.detect(cnts[0])
                    
                    cv2.putText(thresh, str(result), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.imshow("Image", thresh)
                    cv2.waitKey()
                shapes.append(result)

        return shapes


if __name__ == "__main__":
    shapesManager = ShapesManager("example1.npy")
    shapesManager.GetShapes()