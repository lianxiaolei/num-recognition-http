# coding: utf-8

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import random


def read_img(file_name, color_inv_norm=True):
    """
    read a image from local file system
    :param file_name:
    :param color_inv_norm:
    :return:
    """
    img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
    origin = cv2.imread(file_name)

    if color_inv_norm:
        img = 255 - img

        img = remove_back(img, np.ones((5, 5), np.uint8))
        # img = img / 255.0

        img = (img - np.min(img)) / (np.max(img) - np.min(img))

        img[img < 0.30] = 0.0
        img[img > 0] += 0.2
        img[img > 1.] = 1.
    else:
        img[img < 50] = 0
        img = img / 255.0

    return img, origin


def remove_back(img, kernel):
    """

    :param img:
    :param kernel:
    :return:
    """
    plt.imshow(img)
    plt.show()

    mask = cv2.adaptiveThreshold(img, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 7)
    plt.imshow(mask)
    plt.show()

    kernel = np.ones((5, 5), np.uint8)

    img_mask = img * (1 - mask)

    img_mask = cv2.morphologyEx(img_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    img_mask = cv2.morphologyEx(img_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    img_mask[img_mask > 0] = 1
    img[img < 128] = 0
    img = img * img_mask

    return img


def region2ndarray(img, region):
    """
    convert a region of img to ndarray
    :param img:
    :param region:
    :return:
    """
    array = img[region.get_y(): region.get_y() + region.get_height(),
            region.get_x(): region.get_x() + region.get_width()]
    return array


def get_hist(img, axis=0):
    """
    return the hist of img with axis
    :param img:
    :param axis:
    :return:
    """
    return np.sum(img, axis=axis)


def get_region_img(img, region):
    """

    :param img:
    :param region:
    :return:
    """
    return img[
           region.get_y(): region.get_y() + region.get_height(),
           region.get_x(): region.get_x() + region.get_width()]


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


def cut_shape(src_img, node, base_coord=(0, 0), eps=0):
    """
    :param src_img: 原图片
    :param coordinate: 提取图片的坐标
    :return:
    """
    cut_img = src_img[node.y + base_coord[1] - eps: node.ey + base_coord[1] + eps,
              node.x + base_coord[0] - eps: node.ex + base_coord[0] + eps]
    cut_img = cv2.cvtColor(cut_img, cv2.COLOR_BGR2GRAY)

    return cut_img


def save_all_regions(regions, dir_name, layer=0):
    """
    save regions as image
    :param regions:
    :param dir_name:
    :param layer:
    :return:
    """
    parent = dir_name[0].split('/')[0]
    if not os.path.exists(parent):
        os.mkdir(parent)

    for i in dir_name:
        if not os.path.exists(i):
            os.mkdir(i)
    for i, question_region in regions.items():
        if layer in [0, 1]:
            cv2.imwrite('%s/%s.jpg' % (dir_name[0], random.randint(1000, 9999)), question_region.get_img() * 255.0)
            cv2.destroyAllWindows()
        if layer in [0, 2]:
            for j, number_region in question_region.get_sub_regions().items():
                cv2.imwrite('%s/%s.jpg' % (dir_name[1], random.randint(1000, 9999)), number_region.get_img() * 255.0)
                cv2.destroyAllWindows()
