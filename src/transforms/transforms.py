"""
@author : Tien Nguyen
@date   : 2023-05-22
"""
import numpy

from torchvision import transforms

class ImageNormalizer(object):
    def __init__(self):
        pass

    def __call__(self, image):
        image = image / 255.0
        return image

class ToTensor(object):
    def __init__(
            self
        ):
        self.to_tensor = transforms.ToTensor()

    def __call__(
            self, image
        ):
        image = image.astype(numpy.float32)
        tensor = self.to_tensor(image)
        return tensor