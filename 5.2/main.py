import numpy as np
import argparse
import imutils
import cv2
import copy

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

print(OBJECT_PARAMS)

THRESHOLD = 200

def getPiece(inputField:np.array, r:int, c:int):
	field = copy.deepcopy(inputField[r * 200:(r + 1) * 200, c * 200:(c + 1) * 200])

	return field

def getAverageBright(piece: np.array):
	return piece.sum() / piece.size

def getThreshold(piece: np.array):
	averageBright = getAverageBright(piece)

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

def get_winner(field:np.array):
	cv2.imshow("Image", field / np.amax(field))
	
	maxValue = np.amax(field)
	
	field /= np.amax(maxValue)

	gridField = field
	
	for i in range(0, 100):
		gridField = cv2.line(gridField, (0,i*200), (2000,i*200), (255,255,0), 1)
		gridField = cv2.line(gridField, (i*200,0), (i*200,2000), (255,255,0), 1)
	
	cv2.imshow("Image", gridField)
	cv2.waitKey(0)

	for i in range(10):
		for j in range(10):
			print(i, j)
			piece = getPiece(field, i, j)

			piece *= 255
			piece = piece.astype(np.uint8)
			
			threshold = getThreshold(piece)

			print(threshold, getAverageBright(piece))

			# for obj in OBJECT_PARAMS:
			# 	if (obj[0] == [i, j]):
			# 		threshold = obj[1]

			print("Threshold - " + str(threshold))
			
			thresh = cv2.threshold(piece, threshold, 255, cv2.THRESH_BINARY)[1]

			cv2.imshow("Image", thresh)
			cv2.waitKey(0)

			cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
			cnts = imutils.grab_contours(cnts)

			shapeDetector = ShapeDetector(0.16)

			if (len(cnts) > 0):

				result = shapeDetector.detect(cnts[0])

				cv2.putText(thresh, str(result), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
				cv2.imshow("Image", thresh)
				cv2.waitKey()

	return result


if __name__ == "__main__":
	arr = np.load("example2.npy")
	print(get_winner(arr))