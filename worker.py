import os
from typing import Type

import torch
from oba import Obj
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from loader.data import Data
from model import auto_model
from model.auto_model import AutoModel
from utils.early_stop import EarlyStop
from utils.gpu import GPU
from utils.logger import Logger
from utils.loss_recorder import LossRecorder
from utils.smart_printer import printer, Bracket, Color, SmartPrinter


class Worker:
    def __init__(self, config, cuda):
        self.config = config
        self.cuda = cuda
        self.device = self.get_device()

        self.print = printer[('MAIN', Bracket.VERTICAL, Color.CYAN)]
        SmartPrinter.logger = Logger(self.config.model.save.log_path)

        self.data = Data(self.config.data)

        self.model_class = self.get_model_class()
        self.model = self.model_class(
            vocab_size=self.data.vocab_size,
            **Obj.raw(self.config.model.network),
        )  # type: AutoModel

        self.data.process(self.model)
        self.model.to(self.device)

        self.total_epochs = self.config.model.policy.epochs
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            params=self.model.parameters(),
            lr=self.config.model.policy.lr
        )
        self.scheduler = optim.lr_scheduler.LambdaLR(
            optimizer=self.optimizer,
            lr_lambda=lambda epoch: (self.total_epochs - epoch) / self.total_epochs,
        )

        for name, p in self.model.named_parameters():  # type: str, torch.Tensor
            if p.requires_grad:
                self.print(name, p.data.shape)

    def get_device(self):
        if self.cuda == -1:
            return 'cpu'
        if self.cuda is None:
            return GPU.auto_choose(torch_format=True)
        return f'cuda:{self.cuda}'

    def get_model_class(self) -> Type[AutoModel]:
        model_dict = {'CBow': auto_model.CBow, 'SkipGram': auto_model.SkipGram}
        for model_name in model_dict:
            if model_name.upper() == self.config.model.model:
                return model_dict[model_name]
        raise ValueError(f'Model {self.config.model.model} not found')

    def train(self):
        dataset = self.data.sets['train']
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.config.model.policy.batch_size,
            shuffle=True,
            pin_memory=False,
        )

        total_loss = LossRecorder()

        self.model.train()
        self.optimizer.zero_grad()
        for step, batch in tqdm(enumerate(dataloader)):
            labels = batch['outputs'].to(self.device)
            outputs = self.model.forward(
                batch=batch['inputs'].to(self.device)
            )
            if self.model.codebook_layers:
                outputs, quantized_loss = outputs
                loss = self.criterion(outputs, labels) + quantized_loss
                loss_dict = dict(loss=loss.item(), quantized_loss=quantized_loss.item())
            else:
                loss = self.criterion(outputs, labels)
                loss_dict = dict(loss=loss.item())
            loss.backward()

            if (step + 1) % self.config.model.policy.accumulation == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

            total_loss.push(loss_dict)

        return total_loss.summarize()

    def dev(self):
        dataset = self.data.sets['dev']
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.config.model.policy.batch_size,
            shuffle=True,
            pin_memory=True,
        )

        total_loss = LossRecorder()

        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader):
                labels = batch['outputs'].to(self.device)
                outputs = self.model.forward(
                    batch=batch['inputs'].to(self.device)
                )
                if self.model.codebook_layers:
                    outputs, quantized_loss = outputs
                    loss = self.criterion(outputs, labels) + quantized_loss
                    loss_dict = dict(loss=loss.item(), quantized_loss=quantized_loss.item())
                else:
                    loss = self.criterion(outputs, labels)
                    loss_dict = dict(loss=loss.item())

                total_loss.push(loss_dict)

        return total_loss.summarize()

    def attempt_save(self, epoch):
        if (epoch + 1) % self.config.model.policy.save_interval == 0:
            path = os.path.join(self.config.model.save.path, f'epoch_{epoch}.bin')
            torch.save(self.model.state_dict(), path)

    def run(self):
        early_stop = EarlyStop(patience=self.config.model.policy.early_stop)

        for epoch in range(self.config.model.policy.epochs):
            train_loss = self.train()
            dev_loss = self.dev()
            self.print(f'epoch {epoch}, train loss: {train_loss}')
            self.print(f'epoch {epoch},   dev loss: {dev_loss}')

            if early_stop.push(epoch, dev_loss['loss']):
                self.print('early stopped')
                break

            self.scheduler.step()
            self.attempt_save(epoch)
