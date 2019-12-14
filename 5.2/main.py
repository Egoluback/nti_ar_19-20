import numpy as np
import argparse
import imutils
import cv2

def removeContrast(field:np.array):
	print(np.amax(field))
	m=np.amax(field)
	field1=field.copy()
	s=field1.shape
	for i in range(s[0]):
		k=0
		for j in range(1,s[1]):
			o=k
			k=field1[i][j]
			if abs(k-o)<2:
				field1[i][j]=0
				field1[i][j-1]=0
			else:
				field1[i][j]=m
	
	field2=field.copy()
	s=field2.shape
	for j in range(s[1]):
		k=0
		for i in range(1,s[0]):
			o=k
			k=field2[i][j]
			if abs(k-o)<2:
				field2[i][j]=0
				field2[i-1][j]=0
			else:
				field2[i][j]=m
	
	for j in range(s[1]):
		for i in range(1,s[0]):
			if field1[i][j]==m or field2[i][j]==m:
				field[i][j]=m
			else: field[i][j]=0
	
	return field

def get_winner(field:np.array):
    field = removeContrast(field)
	
    field /= np.amax(field)
    image = field
	
    cv2.imshow("Image", image)
    cv2.waitKey(0)

    gray = cv2.cvtColor(field, cv2.COLOR_BGR2GRAY)

    thresh = cv2.threshold(gray, 0.2, 255, cv2.THRESH_BINARY)[1]

    print(thresh)
    
    cv2.imshow("Image", thresh)
    cv2.waitKey(0)

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	# cnts = imutils.grab_contours(cnts)

    return field.shape


if __name__ == "__main__":
	arr = np.load("example1.npy")
	print(get_winner(arr))