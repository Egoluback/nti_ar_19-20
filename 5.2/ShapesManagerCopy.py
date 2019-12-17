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
SIZE = 200

class ShapesManager:
    def __init__(self, filePath):
        self.filePath = filePath

        self.field = np.load(self.filePath)
    
    @private
    def getPiece(self, inputField:np.array, r:int, c:int):
        field = copy.deepcopy(inputField[r * SIZE:(r + 1) * SIZE, c * SIZE:(c + 1) * SIZE])

        return field

    @private
    def getAverageBright(self, piece: np.array):
        return piece.sum() / piece.size

    @private
    def getThreshold(self, piece: np.array):
        averageBright = self.getAverageBright(piece)

        if (averageBright > 0 and averageBright <= 30):
            return 60
        elif (averageBright > 30 and averageBright <= 50):
            return 60
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
        # cv2.imwrite("image.png", cv2.cvtColor(self.field / np.amax(self.field), cv2.COLOR_GRAY2RGB))
        # cv2.imwrite("image.png", self.field / np.amax(self.field))
        
        maxValue = np.amax(self.field)
        
        self.field /= np.amax(maxValue)

        field2 = self.field
        
        field2 *= 255

        for i in range(0, 100):
            field2 = cv2.line(field2, (0,i*200), (2000,i*200), (0,0,0), 1)
            field2 = cv2.line(field2, (i*200,0), (i*200,2000), (0,0,0), 1)

        field2 = field2.astype(np.uint8)

        for i in range(10):
            for j in range(10):
                cv2.floodFill(field2, None, (i * 200 + 1, j * 200 + 1), 0)

        # result = cv2.adaptiveThreshold(field2, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 51, 2)
        result = field2

        # result = cv2.addWeighted(field2, 5, result, -1, 0, result)
        
        # result = cv2.adaptiveThreshold(field2, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 101, 2)

        
        # result = cv2.morphologyEx(result, cv2.MORPH_OPEN,   cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10)))
        
        cv2.imshow("image", result)
        cv2.waitKey(0)

        for i in range(10):
            for j in range(10):
                print(i, j)
                piece = self.getPiece(result, i, j)
                cnts = cv2.findContours(piece.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cnts = imutils.grab_contours(cnts)
                shapeDetector = ShapeDetector(0.045)

                resultShape = 0

                if (len(cnts) > 0):
                    print("Object spotted")

                    resultShape = shapeDetector.detect(cnts[0])
                    
                    if (resultShape == "unidentified"): continue

                    cv2.putText(piece, str(resultShape), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.putText(piece, str(i) + "," + str(j), (100, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.imshow("Image", piece)
                    cv2.waitKey()

        return shapes


if __name__ == "__main__":
    shapesManager = ShapesManager("example2.npy")
    shapesManager.GetShapes()