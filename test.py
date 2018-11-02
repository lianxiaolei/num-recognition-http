
import cv2
import numpy as np
import os
from utils.normalization import chirography_normal
import matplotlib.pyplot as plt

def rand(r):
    base = 256.0
    a = 17.0
    b = 139.0
    tmp0 = a * r + b
    tmp1 = tmp0 // base
    tmp = tmp0 - tmp1 * base
    p = tmp / base
    return p


def move(path=r'F:\123\character'):
    l = os.listdir(path)
    for ll in l:
        print(ll[-3:])
        os.rename(os.path.join(path, ll), os.path.join(path, str(int(ll[-3:]) - 1)))


def dododo(fname):
    img = 255 - cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
    kernel = np.ones((3, 3), np.uint8)
    img = cv2.dilate(img, kernel)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    # img = img / 255.0
    img = cv2.resize(img, (42, 42))

    img = np.pad(img, ((3,), (3,)), mode='constant')
    return img


def norm(path=r'F:\123\character', target='None', inv=False):
    for i in range(0, 10):
        l = os.listdir(os.path.join(path, str(i)))
        print('now in', i)
        index = 0
        for fname in l:
            img = cv2.imread(os.path.join(path, str(i), fname), cv2.IMREAD_GRAYSCALE)
            try:
                if inv:
                    img = 255 - img
            except Exception as e:
                print(fname)
                continue

            img = (img - np.min(img)) / (np.max(img) - np.min(img))

            img = cv2.resize(img, (48, 48))

            img = cv2.GaussianBlur(img, (5, 5), 0)

            img[img < 0.16] = 0

            try:
                img = chirography_normal(img)
            except Exception as e:
                print(e)
                print(fname)
                continue

            # plt.imshow(img)
            if not os.path.exists(os.path.join(target, str(i))):
                os.mkdir(os.path.join(target, str(i)))

            cv2.imwrite(os.path.join(target, str(i), str(i) + '_' + str(index) + '.jpg'), img * 255)
            index += 1


def check(path):
    for i in range(0, 10):
        l = os.listdir(os.path.join(path, str(i)))
        print('now in', i)
        for fname in l:
            img = cv2.imread(os.path.join(path, str(i), fname), cv2.IMREAD_GRAYSCALE)
            try:
                len(img)
            except Exception as e:
                print(os.path.join(path, str(i), fname))
                os.remove(os.path.join(path, str(i), fname))

if __name__ == '__main__':
    # move('f:/123/character')
    # norm('f:/123/new', 'f:/234/new', inv=False)
    # norm('f:/123/NNT_binT', 'f:/234/NNT_binT', inv=False)
    check('f:/num_ocr')