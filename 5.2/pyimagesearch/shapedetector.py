import cv2
import math

class ShapeDetector:
    def __init__(self, accuracy, size):
        #точность для апроксимации фигуры полигоном
        self.accuracy = accuracy
        self.size = size
    
    #расчет расстояния между двух строчек
    def distance(self,p1,p2):
        return ((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)**0.5
        
    def detect(self, c):
        shape = 'unidentified'
        p = cv2.arcLength(c, True)
        a = cv2.approxPolyDP(c, self.accuracy * p, True)
        # хорошая попытка :) но не для выданных тестов. все триги там это трапеции т.е. у тригов 4 вершины
        if (len(a)==3):
            shape = '3'
        elif (len(a) == 4):
            (x,y,w,h) = cv2.boundingRect(a)
            ar = w/h
            shape = '1' # сквеер
            # тестим, что он похож на квадрат по соотношению сторон
            if (abs(ar-1) < 0.05):
                shape = '1'
                return shape

            # если не сквеер, то триг
            # определяем ориентацию трига
            # находим длины сторон полигона
            sides = [
                self.distance(a[0][0],a[1][0]),
                self.distance(a[1][0],a[2][0]),
                self.distance(a[2][0],a[3][0]),
                self.distance(a[3][0],a[0][0])
            ]
            
            #проверяем что это трапеция (триг)
            s_min = min(sides)
            s_max = max(sides)
            if (s_max/s_min-1 > 0.5):
                # ищем минимальную сторону
                min_i = sides.index(s_min)
                # ищем точки, которые ее породили
                p1 = a[(min_i+1)%4][0]
                p2 = a[min_i][0]
                #минимальная сторона слева?
                if (p1[0]<self.size//2 and p2[0]<self.size//2):
                    shape = 6
                #минимальная сторона справа?
                elif (p1[0]>self.size//2 and p2[0]>self.size//2):
                    shape = 4
                #минимальная сторона сверху?
                elif (p1[1]<self.size//2 and p2[1]<self.size//2):
                    shape = 3
                #минимальная сторона снизу?
                elif (p1[1]>self.size//2 and p2[1]>self.size//2):
                    shape = 5
        # не триги и не сквееры это секлы
        else:
            shape = '2'
        return shape