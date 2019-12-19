import numpy as np
import cv2
import copy
import imutils

from pyimagesearch.shapedetector import ShapeDetector
from ShapesManager import ShapesManager
from getWinner import *

SIZE = 200 #размер сегмента игрового поля в пикселях
GAME_W = 10 # ширина игрового поля
GAME_H = 10 # выстоа игрового поля

def get_winer(field:np.array, N:int):
    sm = ShapesManager(field, SIZE)
    game_fld = sm.GetShapes()
    res = winner(game_fld, N, GAME_W, GAME_H)
    print(res)

if __name__ == "__main__":
    arr = np.load("example1.npy")
    get_winer(arr, 15)