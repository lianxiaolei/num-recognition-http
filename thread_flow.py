# coding: utf-8

from core.processor import *
import warnings
from core.img_process import *

warnings.filterwarnings("ignore")


def run(file_name):
    """
    处理主流程
    :param file_name:
    :return:
    """
    img, origin = read_img(file_name, color_inv_norm=True)
    regions = cut(img, row_eps=img.shape[1] / 30, col_eps=10, display=False)
    save_all_regions(regions, dir_name=['data/ques', 'data/nums'])
    keras_recognition(regions, origin, 'model/cnn3_gen.h5')


if __name__ == '__main__':
    run('images/cz1.jpg')

    # from keras.preprocessing.image import ImageDataGenerator
    # dg = ImageDataGenerator()
    # dg.flow_from_directory()
