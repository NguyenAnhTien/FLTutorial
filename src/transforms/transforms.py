"""
@author : Tien Nguyen
@date   : 2023-05-08
"""

class ImageNormalizer(object):
    def __init__(self):
        pass

    def __call__(self, image):
        image = image / 255.0
        return image
