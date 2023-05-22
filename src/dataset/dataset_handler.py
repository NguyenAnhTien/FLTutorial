"""
@author : Tien Nguyen
@date   : 2023-05-21
"""
import numpy

import torch
from torchvision import transforms

import constants
from transforms.transforms import ImageNormalizer

class DatasetHandler(torch.utils.data.Dataset):
    def __init__(
            self,
            images: list,
            labels: list
        ) -> None:
        self.images, self.labels = images, labels
        self.define_transforms()

    def __len__(
            self
        ):
        return len(self.images)

    def __getitem__(
            self, index: int
        ) -> dict:
        image = self.images[index]
        label = self.labels[index]
        image = self.transforms(image)
        return {
            constants.IMAGE : image,
            constants.LABEL : label
        }

    def define_transforms(
            self
        ) -> None:
        self.transforms = transforms.Compose([
            ImageNormalizer(),
        ])
