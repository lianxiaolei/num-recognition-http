# coding: utf-8

import os
import uuid
from core.cnn import CNN
from core.img_process import *
from core.segmentation import *
import os
import cv2
import numpy as np
from keras.models import load_model
from utils.preprocessing import *
from sklearn.model_selection import train_test_split


def cut(img, row_eps, col_eps, display=False):
    """
    cut a image
    :param img:
    :param row_eps:
    :param col_eps:
    :param display:
    :return:
    """
    question_areas = project_cut(img, row_eps, col_eps)
    # show_all_regions(img, question_areas, layer=1)
    for k, v in question_areas.items():
        region_arr = region2ndarray(img, v)

        number_areas = project_cut(
            region_arr, 0, 0, resize=True, display=display)

        v.sub_regions = number_areas

    return question_areas


def save_region_as_jpg(fname, img, region, diastolic=True):
    """

    :param fname:
    :param img:
    :param region:
    :param diastolic:
    :return:
    """
    sub_img = get_region_img(img, region)

    if diastolic:
        cv2.imwrite(fname, sub_img * 255)
    else:
        cv2.imwrite(fname, sub_img)

    cv2.destroyAllWindows()


def get_new_data(base_path):
    """

    :param base_path:
    :return:
    """
    nums = os.listdir(base_path)
    train_data = []
    train_label = []
    lbl = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]
    for num in nums:
        calc = 0
        jpgs = os.listdir(os.path.join(base_path, num))
        print('-' * 30, 'now load %s' % num, '-' * 30)
        for jpg in jpgs:
            # calc += 1
            # if calc > 5000:
            #     print('the %s data is more than 5000' % num)
            # break

            fname = os.path.join(base_path, num, jpg)
            pic = read_img(fname, color_inv_norm=False)
            train_data.append(pic)
            train_label.append(lbl[int(num)])

    train_data = np.array(train_data)
    train_label = np.array(train_label)
    # print(train_data.shape, train_label.shape)
    # print(train_data)
    # print(np.argmax(train_label, axis=1))
    return train_data, train_label


def alg_train_new(model_name, p_keep_conv=1.0, p_keep_hidden=1.0,
                  batch_size=512, test_size=256, epoch_time=3):
    """
    :param model_name:
    :param p_keep_conv:
    :param p_keep_hidden:
    :param batch_size:
    :param test_size:
    :param epoch_time
    :return:
    """
    print('initializing CNN model')
    cnn = CNN(p_keep_conv=p_keep_conv, p_keep_hidden=p_keep_hidden,
              batch_size=batch_size, test_size=test_size, epoch_time=epoch_time)
    print('CNN has been initialized')

    # print('load mnist done')
    print('load training data')
    X, y = get_new_data('F:/num_ocr')
    X = X / 255.0

    X = X.reshape(-1, 48, 48, 1)
    # X = X.reshape(-1, 28, 28, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)
    print('load training data done')
    print('-' * 30, 'training', '-' * 30)
    import time
    tmp_time = time.time()
    cnn.fit_new(X_train, y_train, X_test, y_test)
    print('total time cost:', time.time() - tmp_time)
    cnn.save(model_name)


def num_recognition(img, model):
    """

    :param img:
    :param cnn:
    :return:
    """
    # result = cnn.predict(img)
    # result = cnn.predict_new(img)
    print(np.max(img))
    # img = 255 - img
    # img = (img - np.min(img)) / (np.max(img) - np.min(img))

    img = preprocessing(img)

    # plt.imshow(img)
    # plt.show()

    result = np.argmax(
        model.predict(
            np.expand_dims(np.expand_dims(img, -1), 0)))

    return result


def keras_recognition(regions, origin, model_name):
    """

    :param img:
    :param model:
    :return:
    """
    model = load_model(model_name)
    for i, question_region in regions.items():
        result = list()

        question_x, question_y = question_region.x, question_region.y
        for j, number_region in question_region.get_sub_regions().items():
            # recognize numbers
            # number_region.set_img(cut_shape(origin, number_region,
            #                                 base_coord=(question_region.x, question_region.y),
            #                                 eps=0))
            # plt.imshow(cut_shape(origin, number_region, base_coord=(question_region.x, question_region.y)))
            # plt.show()

            num = num_recognition(number_region.get_img(), model)
            cv2.rectangle(origin, (question_x + number_region.x, question_y + number_region.y),
                          (question_x + number_region.x + number_region.width,
                           question_y + number_region.y + number_region.height),
                          (0, 255, 0), 2)

            if not os.path.exists('result'):
                os.mkdir('result')

            st = str(uuid.uuid1())
            fname = 'result/%s-%s.jpg' % (str(num), st)

            cv2.imwrite(fname, number_region.get_img() * 255)

            font = cv2.FONT_HERSHEY_PLAIN
            cv2.putText(origin, str(num), (question_x + number_region.x, question_y + number_region.y),
                        font, 4, (0, 255, 0), 2)

    cv2.imwrite('result.jpg', origin)


if __name__ == '__main__':
    pass
