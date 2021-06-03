import numpy as np
import cv2, imutils, argparse, copy

from cvHelper_module import CVHelper

ap = argparse.ArgumentParser()

ap.add_argument("--image", "--i", required=True, help = "Path to image.")

args = vars(ap.parse_args())

image = cv2.imread(args["image"])

# initializing CVHelper class; setting default image as an argument
helper = CVHelper(image, 'https://api.arstand-lab.ru/api/0/task/test_task/', '374270759c9852f00f667cb0308d2f8c2f600ec0')
token = "374270759c9852f00f667cb0308d2f8c2f600ec0"
HEADERS = {'Authorization': f'Token {token}'}

cv2.imshow("original", image)
cv2.imshow("field test", helper.createFieldByShapes(image, 5))
cv2.waitKey()

# print(helper.postImageStand(HEADERS, open("image.jpg", "rb"), "default").text)

# grid = helper.getGrid() # making grid
# image_grid = cv2.bitwise_and(image, grid) # adding grid to image using bitwise

# shape = helper.getShapeFromGrid([1, 1], 0, 0, (5, 3)) # getting shape from image grid; indexes, bias and grid size as arguments

# field, params = helper.createCursedGrid([(0, 0), (800, 0), (0, 1000), (800, 1000)], (800, 1000), (8, 10)) # making tight grid; positions of markers(array), background size and cells ratio as arguments; method returns field array and markers params array

# print("Center of 1th cell: " + str(helper.getCellCenter(params, 1, 1))) # getting center of cell; we use params we've got from createCursedGrid method

# cv2.imshow("image with grid", image_grid)
# cv2.imshow("shape", shape)
# cv2.imshow("field", field)
# cv2.waitKey(0)
