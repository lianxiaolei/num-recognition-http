# coding: utf-8

from collections import OrderedDict


class Region(object):
    """
    This class is used to describe a rectangle.
    It has five vars include x, y, width, height and sub_regions,
        the sub_regions is also a Region object.
    """

    def __init__(self, x=0, y=0, width=0, height=0, img=None):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.sub_regions = OrderedDict()
        self.img = img
        self.recognition = None
        self.result = None
        if width and height:
            self.ex = self.x + self.width
            self.ey = self.y + self.height
        # print 'Class has been initialized', vars(self)

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y

    def get_width(self):
        return self.width

    def get_height(self):
        return self.height

    def set_x(self, x):
        self.x = x

    def set_y(self, y):
        self.y = y

    def set_width(self, width):
        self.width = width

    def set_height(self, height):
        self.width = height

    def get_sub_regions(self):
        return self.sub_regions

    def add_sub_region(self, region):
        self.sub_regions.append(region)

    def get_img(self):
        return self.img

    def set_img(self, img):
        self.img = img

    def get_recognition(self):
        return self.recognition

    def set_recognition(self, reco_str):
        self.recognition = reco_str

    def get_result(self):
        return self.result

    def set_result(self, result):
        self.result = result


# test
if __name__ == '__main__':
    r = Region(10, 10, 2, 2)
