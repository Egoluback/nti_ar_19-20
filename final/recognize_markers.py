import requests, cv2, imutils
import numpy as np

from PIL import Image
from skimage.filters import threshold_local
from transform.transform import four_point_transform

URL = 'https://api.arstand-lab.ru'

token  = 'Token 374270759c9852f00f667cb0308d2f8c2f600ec0'
HEADERS = {'Authorization': token, 
                       'Content-Type': 'application/json'}

def get_markers():
    res = requests.get(f'{URL}/api/0/marker/get_markers/', headers=HEADERS)
    if res.ok:
        with open('file.npz', 'wb') as f:
            f.write(res.content)
        return True
    else:
        return False
    
def post_ids(markers):
    content = {'markers': str(markers)}
    res = requests.post(f'{URL}/api/0/marker/check_markers/all/', 
                        headers=HEADERS, 
                        json=content)
    print(res.status_code, res.text)
    if res.ok:
        return True
    else:
        return False

def getCircleRotation(image, image_mask_rotation):
    image_mask_rotation = cv2.cvtColor(image_mask_rotation, cv2.COLOR_BGR2GRAY)
    image_copy = image.copy()
    image_copy = cv2.rectangle(image_copy, (0, 0), (300, 20), (0, 0, 0), -1)
    image_copy = cv2.rectangle(image_copy, (280, 0), (300, 300), (0, 0, 0), -1)
    image_copy = cv2.rectangle(image_copy, (0, 280), (300, 300), (0, 0, 0), -1)
    image_copy = cv2.rectangle(image_copy, (0, 0), (20, 300), (0, 0, 0), -1)

    # ver1 93 -50
    # ver2 73 -50
    image_copy = cv2.adaptiveThreshold(image_copy, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 73, -50) 
    # cv2.imshow("rot thresh", image_copy)
    image_copy = cv2.bitwise_and(image_copy, image_mask_rotation)
    # cv2.imshow("rot and", image_copy)
    image_copy = cv2.bitwise_xor(image_copy, image_mask_rotation)
    # cv2.imshow("rot xor", image_copy)

    sum_nums = (image_copy[:150, :150].sum() // 255, image_copy[: 150, 150 :].sum() // 255, image_copy[150 :, 150:].sum() // 255, image_copy[150 :, : 150].sum() // 255)

    return sum_nums.index(min(sum_nums)) * 90

def get_MaskMarkers():
    return [cv2.threshold(cv2.cvtColor(np.array(Image.open("masks/markers/" + str(i) + ".png")), cv2.COLOR_BGR2GRAY), 50, 255, cv2.THRESH_BINARY)[1] for i in range(7)]

def getWarpedImage(image):
    field = cv2.imread("masks/field.png")
    field = cv2.cvtColor(field, cv2.COLOR_BGR2GRAY)

    # top, left, bottom, right
    average_sides = (round(np.average(image[: 30, : 300])), round(np.average(image[: 300, : 30])), round(np.average(image[270 : 300, : 300])), round(np.average(image[: 300, 270 : 300])))

    normed = np.array(average_sides) / np.array(average_sides).max()

    needMove = True

    for value in normed:
        if (value < 1):
            needMove = needMove and (1 / value > 1.5)
    
    if (needMove):
        moveSideIndex = average_sides.index(np.array(average_sides).max())

        if (moveSideIndex == 0):
            # print("top")
            toMove = image[int(average_sides[moveSideIndex]) // 10 : 300, : 300].copy()
            image[int(average_sides[moveSideIndex]) // 10 : 300, : 300] = (np.ones((300 - int(average_sides[moveSideIndex]) // 10, 300)) * 255).astype(np.uint8)
            image[: 300 - int(average_sides[moveSideIndex]) // 10, : 300] = toMove
        elif (moveSideIndex == 1):
            # print("left")
            toMove = image[: 300, int(average_sides[moveSideIndex]) // 10 : 300].copy()
            image[: 300, int(average_sides[moveSideIndex]) // 10 : 300] = (np.ones((300, 300 - int(average_sides[moveSideIndex]) // 10)) * 255).astype(np.uint8)
            image[: 300, : 300 - int(average_sides[moveSideIndex]) // 10] = toMove
        elif (moveSideIndex == 2):
            # print("bottom")
            toMove = image[: 300 - int(average_sides[moveSideIndex]) // 10, : 300].copy()
            image[: 300 - int(average_sides[moveSideIndex]) // 10, : 300] = (np.ones((300 - int(average_sides[moveSideIndex]) // 10, 300)) * 255).astype(np.uint8)
            image[int(average_sides[moveSideIndex]) // 10 : 300, : 300] = toMove
        elif (moveSideIndex == 3):
            # print("right")
            toMove = image[: 300, : 300 - int(average_sides[moveSideIndex]) // 10].copy()
            image[: 300, : 300 - int(average_sides[moveSideIndex]) // 10] = (np.ones((300, 300 - int(average_sides[moveSideIndex]) // 10)) * 255).astype(np.uint8)
            image[: 300, int(average_sides[moveSideIndex]) // 10 : 300] = toMove

        # cv2.imshow("toMove", image[: 300, : 300 - int(average_sides[moveSideIndex]) // 10])
        # cv2.imshow("moved", image[: 300, int(average_sides[moveSideIndex]) // 10 : 300])
        # cv2.waitKey(0)

    image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 151, -4) # 93 -4

    field[50 : 350, 50 : 350] = image
    image = field

    image_original = image.copy()
    ratio = image.shape[0] / 500
    image = imutils.resize(image, height=500)
    image_copy = image.copy()

    image = cv2.GaussianBlur(image, (3, 3), 0)
    image = cv2.threshold(image, 50, 255, cv2.THRESH_BINARY)[1]
    image = cv2.Canny(image, 10, 255)

    contours = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)[0 : 4]
    screen_contour = None

    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.05 * perimeter, True)
        if (len(approx) == 4):
            screen_contour = approx

    if (screen_contour is None): 
        return None

    image_copy = cv2.drawContours(image_copy, [screen_contour], -1, (255, 0, 0), 1)

    return cv2.resize(four_point_transform(image_original, screen_contour.reshape(4, 2) * ratio), (300, 300))

if __name__ == '__main__':
    # print('GET successful: ', get_markers())

    arr = np.load("file.npz")
    image_mask_rotation = cv2.threshold(np.array(Image.open("masks/mask_all.png")), 50, 255, cv2.THRESH_BINARY)[1]

    markers_image = arr['markers']
    coords_image = arr['coords']

    rotationMarker = 0

    result = []

    for imageIndex in range(len(markers_image)):
        
        # image = cv2.imread("samples/" + str(imageIndex) + ".png")

        image = markers_image[imageIndex]
        # cv2.imshow("original", image)
        masks_markers = get_MaskMarkers()
        
        warped = getWarpedImage(image)
        if (warped is not None):
            image = warped

        rotation = getCircleRotation(image, image_mask_rotation)
        rotationMarker = rotation

        for i in range(rotation // 90):
            image = np.rot90(image.copy())
        # cv2.imshow("rotation", image)

        image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 151, -4) # 93 -4
        
        erodes = []

        for mask in masks_markers:
            image_mask = cv2.bitwise_and(image, mask)
            # cv2.imshow("and", image_mask)
            image_mask = cv2.bitwise_xor(image_mask, mask)
            # cv2.imshow("xor", image_mask)
            
            image_mask = cv2.erode(image_mask, np.ones((9, 9), np.uint8))
            # cv2.imshow("erode", image_mask)

            erodes.append(image_mask.sum() // 255)
            # cv2.waitKey(0)
            
        # cv2.imshow("result", masks_markers[erodes.index(min(erodes))])
        # cv2.waitKey(0)
        result.append([erodes.index(min(erodes)) + 1, coords_image[imageIndex][0], coords_image[imageIndex][1], rotationMarker])

    print(result)
    
    # print('POST rotations successful: ', post_ids(result))