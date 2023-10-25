"""
@author : Tien Nguyen
@date   : 2023-05-21
"""
import argparse

from tqdm import tqdm

import torch

import wandb

import utils
import constants

from network import Model
from client import Client
from configs import Configurer
from server import start_server
from dataset import DatasetHandler

def load_data(configs):
    clients = []
    X_train = utils.read_image_data("train-images.idx3-ubyte")
    y_train = utils.read_labels("train-labels.idx1-ubyte")
    X_test = utils.read_image_data("t10k-images.idx3-ubyte")
    y_test = utils.read_labels("t10k-labels.idx1-ubyte")
    train_dataset = DatasetHandler(images=X_train, labels=y_train)
    val_dataset = DatasetHandler(images=X_test, labels=y_test)

    client_size = X_train.shape[0] // configs.num_clients
    share_data = []
    for index in range(0, client_size * configs.num_clients, client_size):
        share_data.append({
                constants.IMAGE : X_train[index : index + client_size],
                constants.LABEL : y_train[index : index + client_size]
            })
    for index in range(configs.num_clients):
        data = share_data[index]
        clients.append(Client(configs,\
                        name=f'client_{index}',\
                        images=data[constants.IMAGE],\
                        labels=data[constants.LABEL]))
    return clients, train_dataset, val_dataset

def train(configs):
    rounds = 5
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
    wandb.init(
        project="FLTutorial",
    )
    max_val_acc = -1
    report_dir = utils.create_report_dir()
    report_dir = utils.join_path(('logs', report_dir))
    clients, train_dataset, val_dataset = load_data(configs)
    train_data_loader = torch.utils.data.DataLoader(\
                                        dataset=train_dataset,\
                                        batch_size=configs.batch_size,\
                                        shuffle=True,\
                                        num_workers=8, drop_last=True)
    val_data_loader = torch.utils.data.DataLoader(\
                                        dataset=val_dataset,\
                                        batch_size=configs.batch_size,\
                                        shuffle=True,\
                                        num_workers=8, drop_last=True)
    fraction = len(clients[0]) / len(train_dataset)
    global_model = Model(configs)
    for _ in tqdm(range(rounds)):
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
        train_loss, train_acc = global_model.eval(train_data_loader)
        val_loss, val_acc = global_model.eval(val_data_loader)
        history['train']['loss'].append(train_loss)
        history['train']['acc'].append(train_acc)
        history['val']['loss'].append(val_loss)
        history['val']['acc'].append(val_acc)
        wandb.log({
            'train_loss' : train_loss,
            'train_acc' : train_acc,
            'val_loss' : val_loss,
            'val_acc' : val_acc
        })
        if max_val_acc < val_acc:
            file_path = f"{val_acc}_{val_loss}_{train_acc}_{train_loss}.pt"
            utils.join_path((report_dir, file_path))
            torch.save(global_model, file_path)    

def execute_server(args, configs):
    start_server()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--configs', type=str, default='configs.yaml',\
                                                    help='configuration file')
    parser.add_argument('--command', type=str, default='True',\
                                                        help='train model')
    
    args = parser.parse_args()
    configs = Configurer(args.configs)

    eval(args.command)(args, configs)
