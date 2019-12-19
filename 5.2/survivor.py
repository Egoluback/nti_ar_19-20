
import numpy as np
import cv2
import copy
import imutils

SIZE = 200 #размер сегмента игрового поля в пикселях
GAME_W = 10 # ширина игрового поля
GAME_H = 10 # выстоа игрового поля

#логика определения победителя
#список двумерный содержит коды кубитоклобусов (см строку 15)
def winner(d,tm): #d - поле игры, tm - кол-во ходов
    time=tm//3
    # 0=void, 1=sq, 2=ci, 3=tr_up, 4=tr_right, 5=tr_down, 6=tr_left
    # обход возможных траекторий
    def dst(x,y,dir,t):
        if x>=GAME_W or x<0 or y>=GAME_H or y<0 or (d[y][x]==1 and t==time):
            return 0
        k=0
        if d1[y][x]==2:
            k=1
            d1[y][x]=0
        if t==1:
            return k
        for i in dirs[dir]:
            k+=dst(x+i[0],y+i[1],dir,t-1)
        k+=dst(x,y,(dir+1)%4,t-1)
        return k
    dirs=(((1,2),(2,1),(-1,2),(-2,1)),((1,2),(2,1),(1,-2),(2,-1)),((1,-2),(2,-1),(-1,-2),(-2,-1)),((-2,-1),(-1,-2),(-1,2),(-2,1)))
    mx=0
    max=0
    xm=0
    ym=0
    for x in range(GAME_W):
        for y in range(GAME_H):
            if d[y][x]>2:
                d1=copy.deepcopy(d)
                mx=dst(x,y,d[y][x]-3,time)
                if mx>max:
                    max=mx
                    xm=x
                    ym=y
    return (xm,ym)

# класс определения кубитоклобуса
class ShapeDetector:
    def __init__(self, accuracy):
        #точность для апроксимации фигуры полигоном
        self.accuracy = accuracy
    
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
                if (p1[0]<SIZE//2 and p2[0]<SIZE//2):
                    shape = 6
                #минимальная сторона справа?
                elif (p1[0]>SIZE//2 and p2[0]>SIZE//2):
                    shape = 4
                #минимальная сторона сверху?
                elif (p1[1]<SIZE//2 and p2[1]<SIZE//2):
                    shape = 3
                #минимальная сторона снизу?
                elif (p1[1]>SIZE//2 and p2[1]>SIZE//2):
                    shape = 5
        # не триги и не сквееры это секлы
        else:
            shape = '2'
        return shape

#класс выделения фигур из изображения
class ShapesManager:
    def __init__(self, npy_data):
        self.field = npy_data
        self.field /= np.amax(self.field)
    
    #вырезка сегмента поля с заданными координатами
    def getPiece(self, inputField:np.array, r:int, c:int):
        field = copy.deepcopy(inputField[r * SIZE:(r + 1) * SIZE, c * SIZE:(c + 1) * SIZE])
        return field

    #получение поля 10х10 с закодированными кубитоклобусами
    def GetShapes(self):
        shapes = [[0] * 10 for i in range(10)]

        fieldCopy = self.field
        fieldCopy *= 255
        fieldCopy = fieldCopy.astype(np.uint8)

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
                cv2.floodFill(thresholdPiece, None, (SIZE-1,1), 0)
                cv2.floodFill(thresholdPiece, None, (1,SIZE-1), 0)
                cv2.floodFill(thresholdPiece, None, (SIZE-1,SIZE-1), 0)
                morphologyPiece = cv2.morphologyEx(thresholdPiece, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)))
                
                # и вот наконец находим контуры
                cnts = cv2.findContours(morphologyPiece.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cnts = imutils.grab_contours(cnts)
                # запускаем детектор полигонов с точностью 0.04 - выдает квадраты (сквееры), трапеции(триги) и прочее(секлы)
                shapeDetector = ShapeDetector(0.04)

                resultShape = 0
                # если в сегменте что-то есть
                if (len(cnts) > 0):
                    # получаем тип полигона
                    resultShape = shapeDetector.detect(cnts[0])
                    if (resultShape == "unidentified"): continue
                    
                    shapes[i][j] = int(resultShape)
        return shapes

def get_winer(field:np.array, N:int):
    sm = ShapesManager(field)
    game_fld = sm.GetShapes()
    res = winner(game_fld, N)
    print(res)

if __name__ == "__main__":
    arr = np.load("example1.npy")
    get_winer(arr, 15)