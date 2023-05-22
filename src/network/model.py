"""
@author : Tien Nguyen
@date   : 2023-05-21
"""
import torch

import constants

class Model(torch.nn.Module):    
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 20, 7)
        self.conv2 = torch.nn.Conv2d(20, 40, 7)
        self.maxpool = torch.nn.MaxPool2d(2, 2)
        self.flatten = torch.nn.Flatten()
        self.fc = torch.nn.Linear(2560, 10)
        self.relu = torch.nn.functional.relu
        self.track_layers = {
            'conv1': self.conv1, 
            'conv2': self.conv2, 
            'fc': self.fc
        }
    
    def forward(self, x_batch):
        out = self.conv1(x_batch)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.flatten(out)
        out = self.fc(out)
        return out
    
    @torch.no_grad()
    def update_params(self, params):
        for layer_name in params:
            self.track_layers[layer_name].weight.data *= 0
            self.track_layers[layer_name].bias.data *= 0
            self.track_layers[layer_name].bias.data += params[layer_name]['bias']
            self.track_layers[layer_name].weight.data += params[layer_name]['weight']
    
    def get_params(self):
        params = {}
        for layer_name in self.track_layers:
            params[layer_name] = {
                'weight': self.track_layers[layer_name].weight.data, 
                'bias': self.track_layers[layer_name].bias.data
            }
        return params

    @torch.no_grad()
    def eval(
            self, 
            data_loader
        ) -> None:
        accs = []
        losses = []
        for batch in data_loader:
            images = batch[constants.IMAGE]
            labels = batch[constants.LABEL]
            outputs = self(images)
            loss = self.criterion(outputs, labels)
            acc = self.accuracy(outputs, labels)
            losses.append(loss)
            accs.append(acc)
        avg_loss = torch.stack(losses).mean().item()
        avg_acc = torch.stack(accs).mean().item()
        return (avg_loss, avg_acc)
