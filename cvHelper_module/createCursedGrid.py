import cv2
import numpy as np


def createCursedGrid(UL, UR, LL, LR):
    U = (UR[0] - UL[0], UR[1] - UL[1])
    L = (UL[0] - LL[0], UL[1] - LL[1])
    R = (UR[0] - LR[0], UR[1] - LR[1])
    D = (LR[0] - LL[0], LR[1] - LL[1])
    image = np.zeros((800, 1000, 3), np.uint8)
    image += 255
    for i in range(11):
        image = cv2.line(image, (UL[0] + U[0] * i // 10, UL[1] + U[1] * i // 10),
                         (LL[0] + D[0] * i // 10, LL[1] + D[1] * i // 10), (0, 0, 0), 1)
    for i in range(9):
        image = cv2.line(image, (LL[0] + L[0] * i // 8, LL[1] + L[1] * i // 8),
                         (LR[0] + R[0] * i // 8, LR[1] + R[1] * i // 8), (0, 0, 0), 1)
    return cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
def getCellCenter(UL,UR,LL,LR,cell_x,cell_y):
    def cross(line1, line2):
        (x1, y1, x2, y2) = line1
        (x3, y3, x4, y4) = line2
        return int(((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) // (
                (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))), int((
                                                                             (x1 * y2 - y1 * x2) * (y3 - y4) - (
                                                                                 y1 - y2) * (
                                                                                     x3 * y4 - y3 * x4)) // (
                                                                             (x1 - x2) * (y3 - y4) - (y1 - y2) * (
                                                                                 x3 - x4)))
    U = (UR[0] - UL[0], UR[1] - UL[1])
    L = (UL[0] - LL[0], UL[1] - LL[1])
    R = (UR[0] - LR[0], UR[1] - LR[1])
    D = (LR[0] - LL[0], LR[1] - LL[1])
    return cross((UL[0] + U[0] * (cell_x + 0.5) // 10, UL[1] + U[1] * (cell_x + 0.5) // 10,
                   LL[0] + D[0] * (cell_x + 0.5) // 10, LL[1] + D[1] * (cell_x + 0.5) // 10),(LL[0] + L[0] * (cell_y + 0.5) // 8, LL[1] + L[1] * (cell_y + 0.5) // 8,
         LR[0] + R[0] * (cell_y + 0.5) // 8, LR[1] + R[1] * (cell_y + 0.5) // 8))