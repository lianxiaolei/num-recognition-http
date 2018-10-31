# coding: utf-8

import os
import uuid
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def chirography_normal(img):
    """

    :param img:
    :return:
    """
    img = img / 255
    img[img < 0.16] = 0
    mask = img > 0
    mask = mask.astype(np.int)
    # print(np.min(mask), np.max(mask))
    plt.imshow(img)
    plt.show()

    maskl = mask[:, :-1]
    maskr = mask[:, 1:]
    masku = mask[1:]
    maskd = mask[: -1]

    horizon_offset = 1 - maskl == maskr
    vertical_offset = 1 - masku == maskd

    # plt.imshow(horizon_offset)
    # plt.show()
    # plt.imshow(vertical_offset)
    # plt.show()

    final_mask = np.zeros(img.shape)
    masko = np.zeros(img.shape)

    final_mask[: -1] += vertical_offset
    final_mask[:, : -1] += horizon_offset
    final_mask[final_mask > 0] = 1
    plt.imshow(final_mask)
    plt.show()

    mid = (img.shape[0] // 2, img.shape[1] // 2)
    mid = [10, 30]
    dist_field = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if final_mask[i, j] <= 0:
                continue
            distance = np.array([1e3, 1e3, 1e3, 1e3])
            if np.any(final_mask[i, j + 1:]) > 0:
                r = int(np.argwhere(final_mask[i, j + 1:] > 0)[0]) - j
                if r > 1:
                    distance[0] = r
            if np.any(final_mask[i + 1:, j]) > 0:
                d = int(np.argwhere(final_mask[i + 1:, j] > 0)[0]) - i
                if d > 1:
                    distance[1] = d
            if np.any(final_mask[i, : j]) > 0:
                l = i - int(np.argwhere(final_mask[i, : j] > 0)[-1])
                if l > 1:
                    distance[2] = l
            if np.any(final_mask[: i, j]) > 0:
                u = j - int(np.argwhere(final_mask[: i, j] > 0)[-1])
                if u > 1:
                    distance[3] = u

            if np.min(distance == 1e3):
                continue
            x = np.argmin(distance) + 1
            print(i, j, distance, x)

            dist_field[i, j] = distance[x - 1]
            masko[i, j] = x

    print(np.min(dist_field), np.max(dist_field))
    plt.imshow(dist_field)
    plt.show()






if __name__ == '__main__':
    img = cv2.imread(r'C:\Users\Administrator.WIN-SJ36M83FJQ9\Music\84.jpg', cv2.IMREAD_GRAYSCALE)
    chirography_normal(img)