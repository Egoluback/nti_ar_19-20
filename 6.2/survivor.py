import numpy as np
import argparse
import imutils
import cv2
import copy
import math
from accessify import private
'''
На вход программы подаются файлы, лежащие с ней в одной папке, как в примере входных данных.
'''
SIZE = 200 #размер сегмента игрового поля в пикселях
GAME_W = 10 # ширина игрового поля
GAME_H = 10 # высота игрового поля
SQUARE = 1
CIRCLE = 2
TRIG_U = 3
TRIG_R = 4
TRIG_D = 5
TRIG_L = 6
X = 0
Y = 1
trigs = {'1121':TRIG_L,'1112':TRIG_U,'2111':TRIG_R,'1211':TRIG_D}
class ShapeDetector:
	def __init__(self, accuracy):
		#точность для апроксимации фигуры полигоном
		self.accuracy = accuracy
	def detect(self, c, pix):
		shape = 0
		p = cv2.arcLength(c, True)
		a = cv2.approxPolyDP(c, self.accuracy * p, True)
		# обрабатываем тригов
		if (len(a) == 3):
			#счетчики признаков
			x_in_ls = 0 #точки с X в левой половине
			x_in_rs = 0 #точки с X в правой половине
			y_in_ls = 0 #точки с Y в верхней половине
			y_in_rs = 0 #точки с Y в нижней половине
			for p in a:
				if (p[0][X] < SIZE//2) and (abs(SIZE//2 - p[0][X])>SIZE//10):
					x_in_ls += 1
				if (p[0][X] > SIZE//2) and (abs(SIZE//2 - p[0][X])>SIZE//10):
					x_in_rs += 1
				if (p[0][Y] < SIZE//2) and (abs(SIZE//2 - p[0][Y])>SIZE//10):
					y_in_ls += 1
				if (p[0][Y] > SIZE//2) and (abs(SIZE//2 - p[0][Y])>SIZE//10):
					y_in_rs += 1
			# из словаря ориентаций тригов определяем ориентацию собрав счетчики в строку
			shape = trigs[''.join(list(map(str,[x_in_ls,y_in_ls,x_in_rs,y_in_rs])))]
		elif (len(a) == 4):
			#проверяем квадрат
			# square = True
			# eps = SIZE // 30 #разброс координат для сквееров
			# for i in range(len(a)):
			# 	if abs(a[i][0][X]-a[(i+1)%4][0][X]) > eps and abs(a[i][0][Y]-a[(i+1)%4][0][Y]) > eps:
			# 		square = False
			# 		break
			if (pix < 128):
				shape = SQUARE
			else:
				shape = CIRCLE
		return shape
def overlap(t):
	'''
	Определитель затенений
	'''
	# Высота источника света
	LSH = 50
	# Радиус цилиндра
	CLR = 2 
	# Диаметр цилиндра
	CLD = 2 * CLR
	# Межклеточное расстояние
	ICD = 5
	n=len(t)
	cylinders = []
	for i in range(0, n):
		for j in range(n):
			data_line = t[i][j]
			ht=[0,6/5,3/5,9/10,9/10,9/10,9/10]
			if data_line!=0:
				cylinder = (ICD * (j - 4), ICD * (i - 4), CLD * ht[data_line], data_line)
				cylinders.append(cylinder)
	overlaps = []
	for i in range(10):
		(x, y, h, t) = cylinders[i]
		d = math.sqrt(x * x + y * y)
		if d > 0:
			t1 = h * d / (LSH - h)
			shad_r = CLR * LSH / (LSH - h)
			shad_d = d + t1
			shad_d_x = x * shad_d / d
			shad_d_y = y * shad_d / d
			for ni in range(10):
				if ni != i:
					(nx, ny, nh, nt) = cylinders[ni]
					if abs(nx - x) <= ICD and abs(ny - y) <= ICD:
						diff_x = nx - shad_d_x
						diff_y = ny - shad_d_y
						nsd = math.sqrt(diff_x * diff_x + diff_y * diff_y)
						if nsd <= shad_r + CLR:
							if not (nx//ICD+4,ny//ICD+4) in overlaps and nt==2:
								overlaps.append((nx//ICD+4,ny//ICD+4))
	return set(overlaps)
class ShapesManager:
	def __init__(self, npy_data, size):
		self.field = npy_data
		self.field /= np.amax(self.field)
		self.accuracy = 0.075
		self.size = size

	#вырезка сегмента поля с заданными координатами
	def getPiece(self, inputField:np.array, r:int, c:int):
		field = copy.deepcopy(inputField[r * self.size:(r + 1) * self.size, c * self.size:(c + 1) * self.size])
		return field

	def getHeight(self, shapeType):
		if (shapeType == 1):
			return "6D/5"
		elif (shapeType == 2):
			return "3D/5"
		else:
			return "9D/10"

	def ChangeAccuracy(self, val):
		self.accuracy += val

	def GetAccuracy(self):
		return self.accuracy

	#получение поля 10х10 с закодированными кубитоклобусами
	def GetShapes(self):
		shapes = [[0] * 10 for i in range(10)]

		fieldCopy = self.field
		fieldCopy *= 255

		fieldCopy = fieldCopy.astype(np.uint8)

		#перебор сегментов изображения в масштабе игрового поля 10х10

		thresholdPiece = cv2.adaptiveThreshold(fieldCopy, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 95, -5)

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
				# shapeDetector = ShapeDetector(0.04, self.size)
				
				shapeDetector = ShapeDetector(self.accuracy)
				resultShape = 0
				# если в сегменте что-то есть
				if (len(cnts) > 0):
					# получаем тип полигона
					resultShape = shapeDetector.detect(cnts[0], morphologyPiece[100][100])
					
					# morphologyPiece = cv2.drawContours(morphologyPiece, [resultShape[1]], -1, (0,255,0), 1)
					# cv2.putText(morphologyPiece, "(" + str(i) + "," + str(j) + ")", (110, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (64, 64, 64), 1)

					# cv2.imshow(str(resultShape[0]), morphologyPiece)
					# cv2.waitKey()

					# print({"type": int(resultShape[0]), "pos": (i, j), "height": self.getHeight(int(resultShape[0]))})

					# cv2.imshow("image", morphologyPiece)
					# cv2.waitKey(0)

					if (resultShape == "unidentified"): continue
					# shapes[i][j] = int(resultShape[0])
					shapes[i][j] = int(resultShape)
		return shapes
def winner(d):
	'''
	Функция, находящая всех сёклов
	'''
	cords=set()
	for y in range(10):
		for x in range(10):
			if d[y][x]==2:
				cords.add((x,y))
	return cords
def count(f):
	'''
	Использует виннер и оверлап, чтобы убивать затенениями
	'''
	sm=ShapesManager(f,200)
	f1=sm.GetShapes()
	olps=overlap(f1)
	f2=winner(f1)
	return f2-olps

"""
def get_images():
	'''
	Функция для тестовых примеров. Извлекается из комментария в случае множественных файлов
	'''
	st=[]
	i=-1
	k=True
	while k:
		i+=1
		try:
			st.append(np.load(f"step_{i}.npy"))
		except:
			k=False
	return st
"""

def get_images():
	# В следующей строке вписать имя входного файла в функцию load внутри кавычек перед .npy
	return np.load(".npy")

def get_survived(m):
	s={(i//10, i%10) for i in range(100)}
	for i in m:
		s=s&count(i)
	return s
if __name__ == "__main__":
	images=get_images()
	surv = get_survived(images)
	for i in surv:
		print(*i)