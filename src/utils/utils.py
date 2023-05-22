"""
@author : Tien Nguyen
@date   : 2023-05-21
"""
from pathlib import Path
from array import array
import struct

import pickle
import json

import numpy

def read_image_data(
        data_file: str
    ) -> list:
    data_dir = Path("../../data/mnist/")
    with open(data_dir/data_file, "rb") as file:
        _, size, rows, cols = struct.unpack(">IIII", file.read(16))
        image_data = array("B", file.read())   
    images = []
    for i in range(size):
        image = numpy.array(image_data[i * rows * cols:(i + 1) * rows * cols]).reshape(28, 28)
        images.append(image)
    return numpy.array(images)
    
def read_labels(
        label_file: str
    ) -> list:
    data_dir = Path("../../data/mnist/")
    with open(data_dir/label_file, "rb") as file:
        magic, _ = struct.unpack(">II", file.read(8))
        if magic != 2049:
            raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
        labels = numpy.array(array("B", file.read()))
    return labels

def read_txt_file(
        filename: str
    ) -> list:
    with open(filename, 'r') as file:
        data = file.readlines()
    data = [item.strip() for item in data]
    return data

def read_pkl(
        pkl_file
    ):
    """
    @desc:
        - reading a pickle file
    """
    with open(pkl_file, 'rb') as file:
        return pickle.load(file)

def write_pkl(
        pkl_file, data
    ) -> None:
    """
    @desc:
        - store data into a pkl file
    """
    with open(pkl_file, 'wb') as file:
        pickle.dump(data, file)

def write_json(data_dict, json_file):
    """
    @desc:
        - store data into a json file
    """
    with open(json_file, 'w') as file:
        json.dump(data_dict, file, indent=4)

def read_json(
        json_file: str
    ) -> dict:
    """
    @desc:
        - read json file
    """
    with open(json_file, 'r') as file:
        data = json.load(file)
    return data
