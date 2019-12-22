import numpy as np
import argparse
import imutils
import cv2
import copy
import math

# from accessify import private

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
            shape = 3
        elif (len(a) == 4):
            (x,y,w,h) = cv2.boundingRect(a)
            ar = w/h
            shape = 1 # сквеер
            # тестим, что он похож на квадрат по соотношению сторон
            if (abs(ar-1) < 0.05):
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
            shape = 2
        return shape


# Light source height
# Высота источника света
LSH = 50
# Cylinder radius
# Радиус цилиндра
CLR = 2 
# Cylinder diameter
# Диаметр цилиндра
CLD = 2 * CLR
# Intercellular distance
# Межклеточное расстояние
ICD = 5


def make_cyl(t):
	n=len(t)
	cylinders = []
	for i in range(0, n):
		for j in range(n):
			data_line = t[i][j]
			ht=[0,6/5,3/5,9/10,9/10,9/10,9/10]
			# print(data_line)
			if data_line!=0:
				cylinder = (ICD * (data_line['pos'][0] - 4), ICD * (data_line['pos'][1] - 4), CLD * ht[data_line['type']])
				cylinders.append(cylinder)
	return cylinders

def overlap(cylinders):
	overlaps = []
	for i in range(10):
		(x, y, h) = cylinders[i]
		d = math.sqrt(x * x + y * y)
		if d > 0:
			t1 = h * d / (LSH - h)
			shad_r = CLR * LSH / (LSH - h)
			shad_d = d + t1
			shad_d_x = x * shad_d / d
			shad_d_y = y * shad_d / d
			for ni in range(10):
				if ni != i:
					(nx, ny, nh) = cylinders[ni]
					if abs(nx - x) <= ICD and abs(ny - y) <= ICD:
						diff_x = nx - shad_d_x
						diff_y = ny - shad_d_y
						nsd = math.sqrt(diff_x * diff_x + diff_y * diff_y)
						if nsd <= shad_r + CLR:
							if not (x*ICD+4,y*ICD+4) in overlaps:
								overlaps.append((nx*ICD+4,ny*ICD+4))
	return overlaps 



#класс выделения фигур из изображения
class ShapesManager:
	def __init__(self, npy_data, size):
		self.field = npy_data
		self.field /= np.amax(self.field)

		self.size = size

	#вырезка сегмента поля с заданными координатами
	# @private
	def getPiece(self, inputField:np.array, r:int, c:int):
		field = copy.deepcopy(inputField[r * self.size:(r + 1) * self.size, c * self.size:(c + 1) * self.size])
		return field

	# @private
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
def winner(d,tm): # экстерминатус сёклов тригами
	time=tm//3
	# 0=void, 1=sq, 2=ci, 3=tr_up, 4=tr_right, 5=tr_down, 6=tr_left
	# обход возможных траекторий
	def dst(x,y,dir,t):
		if x>=10 or x<0 or y>=10 or y<0 or (d[y][x]==1):
			return 0
		k=0
		if d[y][x]==2:
			k=1
			d[y][x]=0
		if t==1:
			return k
		for i in dirs[dir]:
			k+=dst(x+i[0],y+i[1],dir,t-1)
		k+=dst(x,y,(dir+1)%4,t-1)
		return k
	dirs=(((1,2),(2,1),(-1,2),(-2,1)),((1,2),(2,1),(1,-2),(2,-1)),((1,-2),(2,-1),(-1,-2),(-2,-1)),((-2,-1),(-1,-2),(-1,2),(-2,1)))
	mx=0
	for x in range(10):
		for y in range(10):
			if d[y][x]>2:
				mx=dst(x,y,d[y][x]-3,time)
	return d

def decode(f): # переводчик адового ужаса в хоть сколько понятную виннеру матрицу
	f1=[[0]*10 for i in range(10)]
	for i in range(10):
		for j in range(10):
			if f[i][j]!=0:
				f1[i][j]=f[i][j]['type']
	return f1

def recode(f): # переводчик матрицы чисел в координаты достойнейших сёклов
	f1=[]
	olps=overlap(make_cyl(nt))
	for i in range(10):
		for j in range(10):
			if f[i][j]==2 and not (j,i) in olps:
				f1.append((j,i))
	return f1

nt=[]
def get_survived(field:np.array, N:int):
	global nt
	sm=ShapesManager(field,200)
	nt=sm.GetShapes()
	for i in recode(winner(decode(nt),N)):
		print(*i)

if __name__ == "__main__":
    arr = np.load("example2.npy")
    get_survived(arr,15)