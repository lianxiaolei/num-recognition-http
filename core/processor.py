# coding: utf-8

from core.cnn import CNN
from core.img_process import *
from tensorflow.examples.tutorials.mnist import input_data
import os
import cv2
from sklearn.model_selection import train_test_split


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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)
    print('load training data done')
    print('-' * 30, 'training', '-' * 30)
    import time
    tmp_time = time.time()
    cnn.fit_new(X_train, y_train, X_test, y_test)
    print('total time cost:', time.time() - tmp_time)
    cnn.save(model_name)


def num_recognition(img, cnn):
    """

    :param img:
    :param cnn:
    :return:
    """
    # result = cnn.predict(img)
    result = cnn.predict_new(img)
    return result


def regions_recognition(regions, model_name):
    """

    :param img:
    :param regions:
    :param model_name:
    :return:
    """
    cnn = CNN()
    cnn.load_session(model_name)
    rec = {'0': '0', '1': '1', '2': '2', '3': '3', '4': '4', '5': '5',
           '6': '6', '7': '7', '8': '8', '9': '9'}

    for i, question_region in regions.items():
        question = []
        stem = []
        answer = []
        flag = 0
        for j, number_region in question_region.get_sub_regions().items():
            # recognize numbers
            result = num_recognition(number_region.get_img(), cnn)
            # add the number recognition result to question region
            regions[i].get_sub_regions()[j].set_recognition(rec[str(result[0])])

            question.append(rec[str(result[0])])

            if rec[str(result[0])] == '=':
                flag = 1
                continue

            if flag == 0:
                stem.append(rec[str(result[0])])
            else:
                answer.append(rec[str(result[0])])

            # if rec[str(result[0])] == '=':
            #     flag = 1

        regions[i].set_recognition(''.join(question))

        stem = ''.join(stem)
        answer = ''.join(answer)
        try:
            regions[i].set_result(eval(stem) == answer)
            print('reco', 'stem:', stem, 'answer:', answer, 'result:', eval(stem) == eval(answer))
        except:
            print('error', 'stem:', stem, 'answer:', answer)
            # raise ValueError('the recognition is incorrect')
            pass

    return regions


if __name__ == '__main__':
    pass
