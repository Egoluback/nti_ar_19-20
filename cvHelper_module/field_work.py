import numpy as np
import cv2, imutils, argparse, copy

from cvHelper_module import CVHelper

shape_size = int(input())

ap = argparse.ArgumentParser()

ap.add_argument("--image", "--i", required=True, help = "Path to image.")

args = vars(ap.parse_args())

image = cv2.imread(args["image"])

helper = CVHelper(image, 'https://api.arstand-lab.ru/api/0/task/test_task/', '374270759c9852f00f667cb0308d2f8c2f600ec0')

image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

image_thresh = cv2.adaptiveThreshold(image_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 15)

cv2.imshow("thresh", image_thresh)

contours = cv2.findContours(image_thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

contours = imutils.grab_contours(contours)

field_pos = []

def sortPoints(field_pos):
    x_points = {point[0]:point[1] for point in field_pos}
    y_points = {point[1]:point[0] for point in field_pos}

    x_min = sorted(x_points)[: 2]
    y_min = sorted(y_points)[: 2]

    x_max = sorted(x_points, reverse = True)[: 2]
    y_max = sorted(y_points, reverse = True)[: 2]

    l_top = list(set(map(lambda point: (point, x_points[point]), x_min)) & set(map(lambda point: (y_points[point], point), y_min)))[0]
    r_top = list(set(map(lambda point: (point, x_points[point]), x_max)) & set(map(lambda point: (y_points[point], point), y_min)))[0]
    l_bottom = list(set(map(lambda point: (point, x_points[point]), x_min)) & set(map(lambda point: (y_points[point], point), y_max)))[0]
    r_bottom = list(set(map(lambda point: (point, x_points[point]), x_max)) & set(map(lambda point: (y_points[point], point), y_max)))[0]

    return [l_top, r_top, l_bottom, r_bottom]

def deleteSamePoints(field_pos):
    field_pos = sorted(field_pos, key = lambda point: point[1])
    for pointIndex in range(len(field_pos)):
        if (pointIndex + 1 <= len(field_pos) - 1 and abs(field_pos[pointIndex + 1][0] - field_pos[pointIndex][0]) <= 10 and abs(field_pos[pointIndex + 1][1] - field_pos[pointIndex][1]) <= 10):
            del field_pos[pointIndex]
    
    return field_pos

contours = sorted(contours, key = cv2.contourArea, reverse = True)

for contour in contours:
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

    if (len(approx) == shape_size):
        moments = cv2.moments(contour)
        center = (int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"]))

        if (center not in field_pos): field_pos.append(center)

field_pos = deleteSamePoints(field_pos)
field_pos = sortPoints(field_pos)
print(field_pos)

field, params = helper.createCursedGrid(field_pos, (800, 1000), (10, 8))

cv2.imshow("field", field)
cv2.waitKey(0)