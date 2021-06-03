import requests
import numpy as np
import cv2, random, json

from Recognizer import Recognizer

URL = 'https://api.arstand-lab.ru'
token = 'Token 374270759c9852f00f667cb0308d2f8c2f600ec0'
HEADERS = {'Authorization': token}


def get_faces_npz():
    res = requests.get(f'{URL}/api/0/game/field/',
                       headers=HEADERS)
    if res.ok:
        with open('file.npz', 'wb') as f:
            f.write(res.content)
        return True
    else:
        print(res.text)
        return False

def get_status():
    res = requests.get(f'{URL}/api/0/game/status/',
                       headers=HEADERS)
    if (res.ok):
        return (res.text, True)
    else:
        return (res.text, False)


def post_faces_projections_npz(file_path):
    files = {'answer': open(file_path, 'rb')}
    res = requests.post(f'{URL}/api/0/task/check_task/field_projection',
                        headers=HEADERS, files=files)
    if res.ok:
        return True
    else:
        print(res.text)
        return False


def get_shadow_point(m):
    x, y, z = tuple(m)
    px, py, pz = tuple(projector)
    oz = pz - z
    ox = x - px
    oy = y - py
    ex = ox * pz / oz
    ey = oy * pz / oz
    return np.array([px + ex, py + ey, 0.])


def vec_from_pts(start, end):
    return (end[0] - start[0], end[1] - start[1], end[2] - start[2])


def vec_vec_prod(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def vec_scalar_prod(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def is_side_illuminated(p1, p2, top):
    v1 = vec_from_pts(top, p1)
    v2 = vec_from_pts(top, p2)
    side_ort_vec = vec_vec_prod(v1, v2)
    illum_vec = vec_from_pts(projector, top)
    return (vec_scalar_prod(side_ort_vec, illum_vec) <= 0)


def is_top(p):
    return p[2] != 0


# [y-1,x-1,y+1,x+1]

def project(image, points, points2):
    image_copy = image.copy().astype('uint8')
    matrix = cv2.getAffineTransform(np.array(points).astype('float32'), np.array(points2).astype('float32'))
    img_res = cv2.warpAffine(image_copy, matrix, (2400, 3000), flags=cv2.INTER_LINEAR)
    return img_res


def replace(point, x, y):
    point[0] += (x + .5) * 300
    point[1] += (y + .5) * 300
    return point


def get_rotation(point):
    if point[0] < 0 and point[1] < 0:
        return 0
    elif point[0] > 0 and point[1] < 0:
        return 1
    elif point[0] < 0 and point[1] > 0:
        return 2
    else:
        return 3


def rotate_point(point, rot):
    for i in range(rot):
        point[0], point[1] = -point[1], point[0]
    return point


def refactor(xyz_face, img_face, img_coord, x, y, rotation):
    image = cv2.cvtColor(np.zeros((3000, 2400)).astype('uint8'), cv2.COLOR_GRAY2BGR)
    points = np.array(
        [replace(rotate_point(xyz_face[0], rotation), x, y), replace(rotate_point(xyz_face[1], rotation), x, y),
         replace(rotate_point(xyz_face[2], rotation), x, y), replace(rotate_point(xyz_face[3], rotation), x, y),
         get_shadow_point(replace(rotate_point(xyz_face[4], rotation), x, y))])
    pointz = [[i[0], i[1], i[2]] for i in points]
    points = [[i[0], i[1]] for i in points]
    if is_side_illuminated(pointz[0], pointz[1], pointz[4]):
        image = image | project(img_face[0], img_coord[0].astype('float32'),
                                np.array([points[4], points[0], points[1]]).astype('float32'))
    if is_side_illuminated(pointz[3], pointz[0], pointz[4]):
        image = image | project(img_face[1], img_coord[1].astype('float32'),
                                np.array([points[4], points[3], points[0]]).astype('float32'))
    if is_side_illuminated(pointz[2], pointz[3], pointz[4]):
        image = image | project(img_face[2], img_coord[2].astype('float32'),
                                np.array([points[4], points[2], points[3]]).astype('float32'))
    if is_side_illuminated(pointz[1], pointz[2], pointz[4]):
        image = image | project(img_face[3], img_coord[3].astype('float32'),
                                np.array([points[4], points[1], points[2]]).astype('float32'))
    # cv2.imshow('i',image)#.astype('int8'))
    # cv2.waitKey(0)
    return image

def refactor_one_side(img_face, image, img_coord, xyz_face, loc, rotation):
    img_face = np.array(img_face)
    img_coord = np.array(img_coord)
    xyz_face = np.array(xyz_face)
    x, y = loc
    points = np.array([get_shadow_point(replace(rotate_point(xyz_face[0], rotation), x, y)),
                       get_shadow_point(replace(rotate_point(xyz_face[1], rotation), x, y)),
                       get_shadow_point(replace(rotate_point(xyz_face[2], rotation), x, y))])
    pointz = [[i[0], i[1], i[2]] for i in points]
    points = [[i[0], i[1]] for i in points]
    if is_side_illuminated(pointz[0], pointz[1], pointz[2]):
        d=project(img_face, img_coord.astype('float32'),
                                np.array([points[0], points[1], points[2]]).astype('float32'))
        #image = image | d
        image[d!=0]=d[d!=0]
    return image


TRIANGLE_IMAGE_COORDS = [[150, 0], [300, 300], [0, 300]]
UL = [-148.170731707317, -148.170731707317, 0]
UR = [148.170731707317, -148.170731707317, 0]
LL = [-148.170731707317, 148.170731707317, 0]
LR = [148.170731707317, 148.170731707317, 0]


def project_adequate_pyramid(texture_face, texture_right, texture_back, texture_left, top_coords, pyramid_x, pyramid_y,
                             image, rot):
    loc = (pyramid_x, pyramid_y)
    image = refactor_one_side(texture_face, image, TRIANGLE_IMAGE_COORDS, [top_coords, LR, LL], loc, rot)
    image = refactor_one_side(texture_right, image, TRIANGLE_IMAGE_COORDS, [top_coords, LL, UL], loc, rot)
    image = refactor_one_side(texture_back, image, TRIANGLE_IMAGE_COORDS, [top_coords, UL, UR], loc, rot)
    image = refactor_one_side(texture_left, image, TRIANGLE_IMAGE_COORDS, [top_coords, UR, LR], loc, rot)
    # image=image|refactor(np.array([UL,UR,LR,LL,top_coords]),np.array([texture_side,texture_side,texture_face,texture_side]),np.array([TRIANGLE_IMAGE_COORDS]*4),pyramid_x,pyramid_y,rot)
    return image


def project_beheaded_pyramid(pyramid_x, pyramid_y, image, rot):
    image = refactor_one_side(pink_face, image, TRIANGLE_IMAGE_COORDS, [[0, 35, 300], LR, LL], (pyramid_x, pyramid_y),
                              rot)
    image = refactor_one_side(pink_side, image, TRIANGLE_IMAGE_COORDS, [[0, -35, 300], UL, UR], (pyramid_x, pyramid_y),
                              rot)
    image = refactor_one_side(pink_side, image, TRIANGLE_IMAGE_COORDS, [[35, 0, 300], UR, LR], (pyramid_x, pyramid_y),
                              rot)
    image = refactor_one_side(pink_side, image, TRIANGLE_IMAGE_COORDS, [[-35, 0, 300], LL, UL], (pyramid_x, pyramid_y),
                              rot)
    image = refactor_one_side(pink_residue, image, TRIANGLE_IMAGE_COORDS,
                              [[0, -35, 300], [35, 35, 300], [-35, 35, 300]], (pyramid_x, pyramid_y), 0)
    return image


def project_flattened_mess(mess_x, mess_y, image, rot):
    image = refactor_one_side(blue_face, image, TRIANGLE_IMAGE_COORDS, [[0, 150, 300], LR, LL], (mess_x, mess_y), rot)
    image = refactor_one_side(blue_back, image, TRIANGLE_IMAGE_COORDS, [[0, 150, 300], UL, UR], (mess_x, mess_y), rot)
    image = refactor_one_side(blue_left, image, TRIANGLE_IMAGE_COORDS, [[36, 150, 300], UR, LR], (mess_x, mess_y), rot)
    image = refactor_one_side(blue_right, image, TRIANGLE_IMAGE_COORDS, [[-36, 150, 300], LL, UL], (mess_x, mess_y),
                              rot)
    return image


def project_yellow_pyramid(x, y, image, rotation):
    return project_adequate_pyramid(yellow_face, yellow_right, yellow_left, yellow_left, [75, 75, 150], x, y, image,
                                    rotation)


def project_purple_pyramid(x, y, image, rotation):
    return project_adequate_pyramid(purple_face, purple_side, purple_side, purple_side, [0, 0, 150], x, y, image,
                                    rotation)


def project_red_pyramid(x, y, image, rotation):
    return project_adequate_pyramid(red_face, red_side, red_side, red_side, [0, 75, 150], x, y, image, rotation)


def project_orange_pyramid(x, y, image, rotation):
    return project_adequate_pyramid(orange_face, orange_right, orange_back, orange_left, [0, 73, 300], x, y, image,
                                    rotation)
def post(image):
    URL = 'https://api.arstand-lab.ru'
    token = '374270759c9852f00f667cb0308d2f8c2f600ec0'
    HEADERS = {'Authorization': f'Token {token}'}
    file = {'answer': open(image,'rb')}
    print(file)
    res = requests.post(f'{URL}/api/0/game/field/', headers=HEADERS, files=file)
    print(res, res.content)


if __name__ == '__main__':
    token  = 'Token 374270759c9852f00f667cb0308d2f8c2f600ec0'
    pink_face = cv2.imread('textures/pink_face.png')
    pink_side = cv2.imread('textures/pink_side.png')
    pink_residue = cv2.imread('textures/pink_residue.png')
    blue_face = cv2.imread('textures/blue_face.png')
    blue_back = cv2.imread('textures/blue_back.png')
    blue_right = cv2.imread('textures/blue_right.png')
    blue_left = cv2.imread('textures/blue_left.png')
    yellow_face = cv2.imread('textures/yellow_face.png')
    yellow_right = cv2.imread('textures/yellow_right.png')
    yellow_left = cv2.imread('textures/yellow_left.png')
    purple_face = cv2.imread('textures/purple_face.png')
    purple_side = cv2.imread('textures/purple_side.png')
    red_face = cv2.imread('textures/red_face.png')
    red_side = cv2.imread('textures/red_side.png')
    orange_face = cv2.imread('textures/orange_face.png')
    orange_right = cv2.imread('textures/orange_right.png')
    orange_left = cv2.imread('textures/orange_left.png')
    orange_back = cv2.imread('textures/orange_back.png')
    projector = [1200, 1500, 3000]
    d = {1:project_flattened_mess, 5:project_red_pyramid, 3:project_purple_pyramid, 7:project_yellow_pyramid,
         6:project_beheaded_pyramid}
    #picture = cv2.cvtColor(np.zeros((3000, 2400)).astype('uint8'), cv2.COLOR_GRAY2BGR)
    picture=cv2.imread('textures/field.jpg')

    # wait = True

    # while wait:
    #     status = get_status()
    #     if (status[1]):
    #         print(json.loads(status[0])["status"])
    #         wait = json.loads(status[0])["status"] != 3
        
    recognizer = Recognizer("/api/0/game/field/", "/api/0/game/figures/", token)
    result = recognizer.Run(get_flag=False, post_flag=False)
    # result = recognizer.Run()

    # toReplace = {0:0, 1:1, 2:2, 3:3, 4:4, 5:2, 6:5, 7:4}
    
    # result = list(map(lambda el: [toReplace[el[0]], el[1], el[2], el[3]], result))
    print("Info: " + str(result))

    for i in result:
        t,x,y,r=tuple(i)
        picture=d[t](y,x,picture,r//90+1)
    
    picture = picture[::-1, ::-1]

    # cv2.imshow('i', cv2.resize(picture, (600, 750)))
    cv2.imwrite('img.jpg', picture)
    # post('img.jpg')
    cv2.waitKey(0)