import numpy as np
import argparse
import imutils
import cv2

def get_winer(field:np.array):
    print(np.amax(field))
    field /= np.amax(field)

    image = field
    
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    return field.shape


if __name__ == "__main__":
    arr = np.load("example1.npy")
    # print(arr)
    print(get_winer(arr))