import cv2, requests, imutils
import numpy as np

class CVHelper:
    def __init__(self, image, stand_url, token):
        self.image = image
        self.stand_url = stand_url
        self.token = token

    def getGrid(self, image = None, grid_size = 50):
        if not image: image = self.image

        grid = image.copy()

        grid = cv2.rectangle(grid, (0, 0), grid.shape[:2][::-1], (255, 255, 255), -1)

        for i in range(100):
            grid = cv2.line(grid, (0, i * grid_size), (1000, i * grid_size), (0, 0, 0), 1)
            grid = cv2.line(grid, (i * grid_size, 0), (i * grid_size, 1000), (0, 0, 0), 1)

        return grid

    def getShapeFromGrid(self, indexes, eH, eW, gridSize, image = None):
        if not image: image = self.image

        (h, w) = image.shape[:2]

        sizeW = w // gridSize[0]
        sizeH = h // gridSize[1]

        return image[sizeH * indexes[0] - eH:sizeH * (indexes[0] + 1) - eH, sizeW * indexes[1] - eW:sizeW * (indexes[1] + 1) - eW]
    
    def createCursedGrid(self, markers_params, size = (800, 1000), ratio = (10, 8)):
        # UL, UR, LL, LR
        U = (markers_params[1][0] - markers_params[0][0], markers_params[1][1] - markers_params[0][1])
        L = (markers_params[0][0] - markers_params[2][0], markers_params[0][1] - markers_params[2][1])
        R = (markers_params[1][0] - markers_params[3][0], markers_params[1][1] - markers_params[3][1])
        D = (markers_params[3][0] - markers_params[2][0], markers_params[3][1] - markers_params[2][1])
        image = np.zeros((size[0], size[1], 3), np.uint8)
        image += 255
        for i in range(ratio[0] + 1):
            image = cv2.line(image, (markers_params[0][0] + U[0] * i // ratio[0], markers_params[0][1] + U[1] * i // ratio[0]),
                            (markers_params[2][0] + D[0] * i // ratio[0], markers_params[2][1] + D[1] * i // ratio[0]), (0, 0, 0), 1)
        for i in range(ratio[1] + 1):
            image = cv2.line(image, (markers_params[2][0] + L[0] * i // ratio[1], markers_params[2][1] + L[1] * i // ratio[1]),
                            (markers_params[3][0] + R[0] * i // ratio[1], markers_params[3][1] + R[1] * i // ratio[1]), (0, 0, 0), 1)
        return (image, markers_params)
    
    def getCellCenter(self, markers_params, cell_x, cell_y):
        def cross(line1, line2):
            (x1, y1, x2, y2) = line1
            (x3, y3, x4, y4) = line2
            return int(((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) // ((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))), int(((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) // ((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)))
        U = (markers_params[1][0] - markers_params[0][0], markers_params[1][1] - markers_params[0][1])
        L = (markers_params[0][0] - markers_params[2][0], markers_params[0][1] - markers_params[2][1])
        R = (markers_params[1][0] - markers_params[3][0], markers_params[1][1] - markers_params[3][1])
        D = (markers_params[3][0] - markers_params[2][0], markers_params[3][1] - markers_params[2][1])
        return cross((markers_params[0][0] + U[0] * (cell_x + 0.5) // 10, markers_params[0][1] + U[1] * (cell_x + 0.5) // 10,
                    markers_params[2][0] + D[0] * (cell_x + 0.5) // 10, markers_params[2][1] + D[1] * (cell_x + 0.5) // 10),(markers_params[2][0] + L[0] * (cell_y + 0.5) // 8, markers_params[2][1] + L[1] * (cell_y + 0.5) // 8,
            markers_params[3][0] + R[0] * (cell_y + 0.5) // 8, markers_params[3][1] + R[1] * (cell_y + 0.5) // 8))

    def setDefaultImage(self, image):
        self.image = image
    
    def postImageStand(self, headers, image, mode = "default"):
        # MODES:
        # default: unit request; returns res
        # line: cyclic requests to stand; returns res and new request with place in line
        
        res = requests.post(f'{self.stand_url}', headers=headers, files={'answer': image})
        if (res.status_code != 200 or mode == "default"): return res
        return (res, requests.post(f'{self.stand_url}', headers=headers, files={'answer': image}))
    
    def createFieldByShapes(self, image, shapeCorners, params = {}):
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_thresh = cv2.adaptiveThreshold(image_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 15)

        contours = cv2.findContours(image_thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        contours = sorted(contours, key = cv2.contourArea, reverse = True)

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

        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

            if (len(approx) == shapeCorners):
                moments = cv2.moments(contour)
                center = (int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"]))

                if (center not in field_pos): field_pos.append(center)

        field_pos = deleteSamePoints(field_pos)
        field_pos = sortPoints(field_pos)

        field, params = self.createCursedGrid(field_pos, (800, 1000), (10, 8))

        return field