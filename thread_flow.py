# coding: utf-8

from core.processor import *
import warnings

warnings.filterwarnings("ignore")


def run(file_name):
    """
    处理主流程
    :param file_name:
    :return:
    """
    # img = read_img(file_name, color_inv_norm=True)
    # regions = cut(img, row_eps=img.shape[1] / 30, col_eps=10, display=True)
    # regions_recognition(regions, 'new_model/Test_CNN_Model.ckpt')

    alg_train_new('model_new/New_CNN_Model.ckpt', epoch_time=10, p_keep_conv=0.8, p_keep_hidden=1.0, batch_size=128)


if __name__ == '__main__':
    run('images/mine9.jpg')

    # from keras.preprocessing.image import ImageDataGenerator
    # dg = ImageDataGenerator()
    # dg.flow_from_directory()
