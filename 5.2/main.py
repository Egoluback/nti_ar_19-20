import numpy as np
import argparse
import imutils
import cv2
import copy

from pyimagesearch.shapedetector import ShapeDetector

THRESHOLD = 50

def removeContrast(inputField:np.array, r:int, c:int):
	field = copy.deepcopy(inputField[r * 200:(r + 1) * 200, c * 200:(c + 1) * 200])

	return field

def get_winner(field:np.array):
	cv2.imshow("Image", field / np.amax(field))
	
	maxValue = np.amax(field)
	
	field /= np.amax(maxValue)

	for i in range(10):
		for j in range(10):
			print(i, j)
			piece = removeContrast(field, i, j)

			piece *= 255
			piece = piece.astype(np.uint8)

			thresh = cv2.threshold(piece, THRESHOLD, 255, cv2.THRESH_BINARY)[1]

			cv2.imshow("Image", thresh)
			cv2.waitKey(0)

			cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
			cnts = imutils.grab_contours(cnts)

			shapeDetector = ShapeDetector(0.1)

			if (len(cnts) > 0):

				result = shapeDetector.detect(cnts[0])

				cv2.putText(thresh, str(result), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
				cv2.imshow("Image", thresh)
				cv2.waitKey()

	return result


if __name__ == "__main__":
	arr = np.load("example1.npy")
	print(get_winner(arr))