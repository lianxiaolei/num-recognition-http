# coding: utf-8

import os
import uuid
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def get_non0_index_scope(l):
    """

    :param l:
    :return:
    """
    if l[0]:
        start = 0
    else:
        start = l.index(True)
    l.reverse()
    if l[0]:
        end = 0
    else:
        end = l.index(True)
    end = len(l) - end
    return start, end


def get_min_content_area(img):
    """

    :param img:
    :return:
    """
    col_proj = (np.sum(img, axis=0) != 0).tolist()
    row_proj = (np.sum(img, axis=1) != 0).tolist()
    col_start, col_end = get_non0_index_scope(col_proj)
    row_start, row_end = get_non0_index_scope(row_proj)

    return row_start, row_end, col_start, col_end


def get_resize_padding_img(img, size=None, padding=None):
    """

    :param img:
    :param size:
    :param padding:
    :return:
    """
    if size and padding:
        sub_img = cv2.resize(img, size)
        sub_img = np.pad(sub_img, padding, mode='constant')
        sub_img = np.pad(sub_img, ((3,), (3,)), mode='constant')
    else:
        sub_img = cv2.resize(img, (28, 28))
    return sub_img


def preprocessing(img):
    row_start, row_end, col_start, col_end = get_min_content_area(img)
    sub_img = img[row_start: row_end, col_start: col_end]

    if col_end - col_start < row_end - row_start:  # 铅直边较长
        change_rate = (row_end - row_start - 42) / float((row_end - row_start))
        changed_width = int((col_end - col_start) * (1 - change_rate))

        if changed_width % 2 == 1:
            changed_width += 1
        if changed_width == 0:
            changed_width = 2
        pad = (42 - changed_width) / 2
        padding = ((0,), (int(pad),))

        #         print(y1 - y0, x1 - x0, 1 - change_rate, changed_width, pad)
        #         plt.imshow(sub_img)
        #         plt.show()

        sub_img = get_resize_padding_img(sub_img, size=(changed_width, 42), padding=padding)

    else:  # 水平边较长
        # change_rate = (x1 - x0 - 42) / float((x1 - x0))
        # changed_height = int((y1 - y0) * (1 - change_rate))

        change_rate = (col_end - col_start - 42) / float((col_end - col_start))
        changed_height = int((row_end - row_start) * (1 - change_rate))

        if changed_height % 2 == 1:
            changed_height += 1
        if changed_height == 0:
            changed_height = 2
        pad = (42 - changed_height) / 2
        padding = ((int(pad),), (0,))

        #         print(y1 - y0, x1 - x0, 1 - change_rate, changed_height, pad)
        #         plt.imshow(sub_img)
        #         plt.show()

        sub_img = get_resize_padding_img(sub_img, size=(42, changed_height), padding=padding)

    return sub_img


def chirography_normal(img, wid=2, eps=0, delta=5):
    """

    :param wid:
    :param eps:
    :param delta:
    :param img:
    :return:
    """

    mask = img > 0
    mask = mask.astype(np.int)
    print(np.min(mask), np.max(mask))
    plt.imshow(img)
    plt.title('origin')
    plt.show()

    maskl = mask[:, :-1]
    maskr = mask[:, 1:]
    masku = mask[1:]
    maskd = mask[: -1]

    horizon_offset = 1 - maskl == maskr
    vertical_offset = 1 - masku == maskd

    final_mask = np.zeros(img.shape)
    masko = np.zeros(img.shape)

    final_mask[: -1] += vertical_offset
    final_mask[:, : -1] += horizon_offset
    final_mask[final_mask > 0] = 1
    # plt.imshow(final_mask)
    # plt.title('final mask')
    # plt.show()

    dist_field = np.zeros(img.shape)

    # plt.imshow(mask)
    # plt.title('mask')
    # plt.show()

    mask[final_mask > 0] = 0
    # plt.imshow(mask)
    # plt.title('mask1')
    # plt.show()

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if final_mask[i, j] <= 0:
                continue
            distance = np.array([1e3, 1e3, 1e3, 1e3])

            r = np.argwhere(mask[i, j + 1:] == 0).reshape(-1, )[0]
            d = np.argwhere(mask[i + 1:, j] == 0).reshape(-1, )[0]
            l = j - np.argwhere(mask[i, : j] == 0).reshape(-1, )[-1]
            u = i - np.argwhere(mask[: i, j] == 0).reshape(-1)[-1]

            if r > eps and np.any(mask[i, j + 1:] > 0) and mask[i, j + 1] > 0:
                distance[0] = r
            if d > eps and np.any(mask[i + 1:, j] > 0) and mask[i + 1, j] > 0:
                distance[1] = d
            if l > eps and np.any(mask[i, : j] > 0) and mask[i, j - 1] > 0:
                distance[2] = l
            if u > eps and np.any(mask[: i, j] > 0) and mask[i - 1, j] > 0:
                distance[3] = u

            x = np.argmin(distance) + 1
            # print(i, j, distance, x)

            if np.min(distance == 1e3):
                # print(i, j, distance, x)
                continue

            dist_field[i, j] = distance[x - 1]
            masko[i, j] = x

            # c = ['green', 'blue', 'red', 'cyan']
            # plt.scatter(j, i, c=c[x - 1])

    # print(np.min(dist_field), np.max(dist_field))
    # plt.imshow(dist_field)
    # plt.title('distance')
    # plt.show()

    for flag in range(1, 5):
        y, x = np.where(masko == flag)
        z = list(zip(y, x))
        # print(z)

        if flag == 1:
            for y, x in z:
                # print(dist_field[y, x] - wid)
                if dist_field[y, x] - wid > delta:
                    mask[y, x: int(x + wid)] = 0
        elif flag == 2:
            for y, x in z:
                if dist_field[y, x] - wid > delta:
                    mask[y: int(y + wid), x] = 0
        elif flag == 3:
            for y, x in z:
                if dist_field[y, x] - wid > delta:
                    mask[y, int(x - wid): x] = 0
        elif flag == 4:
            for y, x in z:
                if dist_field[y, x] - wid > delta:
                    mask[int(y - wid): y, x] = 0

    # plt.imshow(mask)
    # plt.title('result mask')
    # plt.show()

    img1 = img * mask
    img1 = cv2.GaussianBlur(img1, (3, 3), 0)
    img1[img1 < 0.16] = 0
    plt.imshow(img1)
    plt.title('result')
    plt.show()

    return img1

    # cv2.imwrite('1.jpg', img1 * 255)


if __name__ == '__main__':
    # img = cv2.imread(r'C:\Users\Administrator.WIN-SJ36M83FJQ9\Music\84.jpg', cv2.IMREAD_GRAYSCALE)
    # img = cv2.imread(r'F:\OCR_SINGLE\new\2\2_7.bmp', cv2.IMREAD_GRAYSCALE)
    # img = cv2.imread(r'F:\OCR_SINGLE\new\2\2_13.bmp', cv2.IMREAD_GRAYSCALE)
    img = cv2.imread(r'F:\OCR_SINGLE\new\8\8_0.bmp', cv2.IMREAD_GRAYSCALE)
    img = cv2.imread(r'F:\OCR_SINGLE\new\2\2_304.bmp', cv2.IMREAD_GRAYSCALE)
    img = cv2.imread(r'F:\OCR_SINGLE\new\0\0_0.bmp', cv2.IMREAD_GRAYSCALE)
    # img = cv2.imread(r'F:\OCR_SINGLE\NNT_binT\5\5_440.jpg', cv2.IMREAD_GRAYSCALE)
    # img = cv2.imread(r'F:\OCR_SINGLE\NNT_binT\5\5_313.jpg', cv2.IMREAD_GRAYSCALE)

    img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-6)
    img[img < 0.16] = 0

    sub_img = preprocessing(img)
    # plt.imshow(sub_img)
    # plt.title('origin')
    # plt.show()
    sub_img = chirography_normal(sub_img, wid=3, delta=3)

