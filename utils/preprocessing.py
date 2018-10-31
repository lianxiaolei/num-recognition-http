# coding:utf8

import numpy as np
import cv2


def preprocessing(img):
    """

    :param img:
    :return:
    """

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

    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = cv2.GaussianBlur(img, (3, 3), 0)

    kernel = np.ones((3, 3), np.uint8)
    img = cv2.erode(img, kernel, iterations=1)

    img = (img - np.min(img)) / (np.max(img) - np.min(img))

    img[img < 0.16] = 0.0

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
        sub_img = get_resize_padding_img(sub_img, size=(changed_width, 42), padding=padding)

    else:  # 水平边较长
        change_rate = (col_end - col_start - 42) / float((col_end - col_start))
        changed_height = int((row_end - row_start) * (1 - change_rate))

        if changed_height % 2 == 1:
            changed_height += 1
        if changed_height == 0:
            changed_height = 2
        pad = (42 - changed_height) / 2
        padding = ((int(pad),), (0,))
        sub_img = get_resize_padding_img(sub_img, size=(42, changed_height), padding=padding)

    return sub_img