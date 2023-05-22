"""
@author : Tien Nguyen
@date   : 2023-05-21
"""
import argparse

import utils
import constants

from network import Model
from client import Client
from configs import Configurer
from dataset import DatasetHandler

def load_data(num_clients):
    clients = []
    X_train = utils.read_image_data("train-images.idx3-ubyte")
    y_train = utils.read_labels("train-labels.idx1-ubyte")
    X_test = utils.read_image_data("t10k-images.idx3-ubyte")
    y_test = utils.read_labels("t10k-labels.idx1-ubyte")
    client_size = X_train.shape[0] // num_clients
    share_data = []
    for index in range(0, client_size * num_clients, client_size):
        share_data.append({
                constants.IMAGE : X_train[index : index + client_size],
                constants.LABEL : y_train[index : index + client_size]
            })
    for index in range(num_clients):
        data = share_data[index]
        clients.append(Client(configs,\
                        name=f'client_{index}',\
                        images=data[constants.IMAGE],\
                        labels=data[constants.LABEL]))
    train_dataset = DatasetHandler(images=X_train, labels=y_train)
    val_dataset = DatasetHandler(images=X_test, labels=y_test)
    return clients, train_dataset, val_dataset

def train(configs):
    rounds = 5
    num_clients = 5
    history = {
        'train' : {
            'loss' : [],
            'acc'  : []
        },
        'val' : {
            'loss' : [],
            'acc'  : []
        }
    }
    clients, train_dataset, val_dataset = load_data(num_clients=num_clients)
    fraction = len(client[0]) / len(train_dataset)
    global_model = Model()
    for _ in rounds:
        new_params = []
        current_params = global_model.get_params()
        for client in clients:
            client.update_params(current_params)
            client.fit()
            params = client.get_params()
            scaled_params = utils.scale_params(params, fraction)
            new_params.append(scaled_params)
        sum_params = utils.sum_weights(new_params)
        global_model.update_params(sum_params)
        train_loss, train_acc = global_model.eval(train_dataset)
        val_loss, val_acc = global_model.eval(val_dataset)
        history['train']['loss'].append(train_loss)
        history['train']['acc'].append(train_acc)
        history['val']['loss'].append(val_loss)
        history['val']['acc'].append(val_acc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--configs', type=str, default='configs.yaml',\
                                                    help='configuration file')
    
    args = parser.parse_args()
    configs = Configurer(args.configs)

    train(configs)
