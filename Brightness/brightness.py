import cv2
import numpy as np


def contract(imgBase, alpha, beta, gamma=0):
    result = cv2.addWeighted(imgBase, alpha, np.ones(imgBase.shape, imgBase.dtype), beta, gamma)
    return result


def changeBrightness(imgBase):
    threshold = 3

    img = cv2.resize(imgBase, (120, 120))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    point = []
    ####Vong ngoai
    point.append(img_gray[14][29])
    point.append(img_gray[44][29])
    point.append(img_gray[74][29])
    point.append(img_gray[104][29])
    point.append(img_gray[104][59])
    point.append(img_gray[104][89])
    point.append(img_gray[74][89])
    point.append(img_gray[44][89])
    point.append(img_gray[14][89])
    point.append(img_gray[14][59])
    ####Vong trong
    point.append(img_gray[29][39])
    point.append(img_gray[59][39])
    point.append(img_gray[89][39])
    point.append(img_gray[89][79])
    point.append(img_gray[59][79])
    point.append(img_gray[29][79])

    summary = sum(point)
    ava = summary / 16

    for i in point:
        if i < ava / threshold or i > ava * threshold:
            point.remove(i)

    summary = sum(point)
    ava = summary / len(point)

    limit = []
    min_2, max_2 = 0, 255
    for i in range(1, 5):
        limit.append(50 * i)
    if min_2 <= ava < limit[0]:
        return contract(imgBase, 1, 100)
    elif limit[0] <= ava < limit[1]:
        return contract(imgBase, 1, 50)
    elif limit[1] <= ava < limit[2]:
        return imgBase
    elif limit[2] <= ava < limit[3]:
        return contract(imgBase, 1, -50)
    else:
        return contract(imgBase, 1, -100)
