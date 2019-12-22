import numpy as np
import argparse
import imutils
import cv2
import copy

from accessify import private

from pyimagesearch.shapedetector import ShapeDetector

#класс выделения фигур из изображения
class ShapesManager:
    def __init__(self, npy_data, size):
        self.field = npy_data
        self.field /= np.amax(self.field)
        
        self.size = size
    
    #вырезка сегмента поля с заданными координатами
    @private
    def getPiece(self, inputField:np.array, r:int, c:int):
        field = copy.deepcopy(inputField[r * self.size:(r + 1) * self.size, c * self.size:(c + 1) * self.size])
        return field
    
    @private
    def getHeight(self, shapeType):
        if (shapeType == 1):
            return "6D/5"
        elif (shapeType == 2):
            return "3D/5"
        else:
            return "9D/10"

    #получение поля 10х10 с закодированными кубитоклобусами
    def GetShapes(self):
        shapes = [[0] * 10 for i in range(10)]

        fieldCopy = self.field
        fieldCopy *= 255
        fieldCopy = fieldCopy.astype(np.uint8)

        cv2.imshow("image", fieldCopy)
        cv2.waitKey()

        #перебор сегментов изображения в масштабе игрового поля 10х10
        for i in range(10):
            for j in range(10):
                #очередной сегмент
                piece = self.getPiece(fieldCopy, i, j)
                # пройдемся адаптивной бинаризацией с magicnumbers 95 и -5 :) для удаления засветов 
                thresholdPiece = cv2.adaptiveThreshold(piece, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 95, -5)
                # устраиваем морфологией перлхарбор мусору
                morphologyPiece = thresholdPiece
                #контрольный в голову. т.к. всяческая контра может засесть по углам и сидеть там в засаде
                cv2.floodFill(thresholdPiece, None, (1,1), 0)
                cv2.floodFill(thresholdPiece, None, (self.size-1,1), 0)
                cv2.floodFill(thresholdPiece, None, (1,self.size-1), 0)
                cv2.floodFill(thresholdPiece, None, (self.size-1,self.size-1), 0)
                morphologyPiece = cv2.morphologyEx(thresholdPiece, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)))
                
                # и вот наконец находим контуры
                cnts = cv2.findContours(morphologyPiece.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cnts = imutils.grab_contours(cnts)
                # запускаем детектор полигонов с точностью 0.04 - выдает квадраты (сквееры), трапеции(триги) и прочее(секлы)
                shapeDetector = ShapeDetector(0.04, self.size)

                resultShape = 0
                # если в сегменте что-то есть
                if (len(cnts) > 0):
                    # получаем тип полигона
                    resultShape = shapeDetector.detect(cnts[0])
                    if (resultShape == "unidentified"): continue
                    
                    shapes[i][j] = {"type": int(resultShape), "pos": (i, j), "height": self.getHeight(int(resultShape))}
        return shapes