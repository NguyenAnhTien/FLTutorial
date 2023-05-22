"""
@author : Tien Nguyen
@date   : 2023-05-21
"""
import torch
import torchmetrics

import constants
from network import Model
from dataset import DatasetHandler

class Client(object):
    def __init__(
            self,
            configs,
            name: str,
            data_loader
        ) -> None:
        self.name = name
        self.configs = configs
        self.data_loader = data_loader
        self.define_model()
        self.train_dataloader()
        self.define_criterion()
        self.configure_optimizers()

    def fit(
            self
        ) -> None:
        history = []
        for _ in range(self.configs.epochs):
            accs = []
            losses = []
            for batch in self.data_loader:
                images = batch[constants.IMAGE]
                labels = batch[constants.LABEL]
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                acc = self.accuracy(outputs, labels)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                loss.detach()
                losses.append(loss)
                accs.append(acc)
            mean_loss = torch.stack(losses).mean().item()
            mean_acc = torch.stack(accs).mean().item()
            history.append((mean_loss, mean_acc))
        return history

    def __len__(
            self
        ) -> int:
        return len(self.dataset_handler)

    def update_params(
            self,
            params: dict
        ) -> None:
        self.model.update_params(params)

    def get_params(
            self
        ) -> dict:
        return self.model.get_params()
    
    def train_dataloader(
            self
        ) -> None:
        self.data_loader = torch.utils.data.DataLoader(\
                                        dataset=self.dataset_handler,\
                                        batch_size=self.configs.batch_size,\
                                        shuffle=True,\
                                        num_workers=8, drop_last=True)
    
    def configure_optimizers(
            self
        ) -> None:
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.configs.lr)
    
    def define_criterion(
            self
        ) -> None:
        self.criterion = torch.nn.CrossEntropyLoss()

    def define_metrics(
            self    
        ) -> None:
        self.accuracy = torchmetrics.Accuracy(task="binary",\
                                        num_classes=self.configs.num_classes)


    def define_model(
            self
        ) -> None:
        self.model = Model()

