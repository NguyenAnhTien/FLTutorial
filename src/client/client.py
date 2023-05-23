"""
@author : Tien Nguyen
@date   : 2023-05-21
"""
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torchmetrics

import constants
from network import Model
from dataset import DatasetHandler

class Client(object):
    def __init__(
            self,
            configs,
            name: str,
            images: list,
            labels: list
        ) -> None:
        self.name = name
        self.configs = configs
        self.setup(images, labels)
        self.define_model()
        self.train_dataloader()
        self.configure_optimizers()

    def fit(
            self
        ) -> None:
        history = []
        for _ in tqdm(range(self.configs.epochs)):
            accs = []
            losses = []
            for batch in self.data_loader:
                images = batch[constants.IMAGE]
                labels = batch[constants.LABEL]
                logits = self.model(images)
                preds = F.softmax(logits, dim=1)
                preds = torch.argmax(preds, dim=1)
                loss = self.model.criterion(logits, labels)
                acc = self.model.accuracy(preds, labels)
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
        self.optimizer = torch.optim.Adam(self.model.parameters(),\
                                                        lr=self.configs.lr)

    def define_model(
            self
        ) -> None:
        self.model = Model(self.configs)

    def setup(
            self,
            images: list,
            labels: list
        ) -> None:
        self.dataset_handler = DatasetHandler(images=images, labels=labels)
