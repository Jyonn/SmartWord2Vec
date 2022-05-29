from typing import Type

from torch.utils.data import DataLoader
from tqdm import tqdm

from loader.data import Data
from model import auto_model
from model.auto_model import AutoModel
from utils.gpu import GPU


class Worker:
    def __init__(self, config, cuda):
        self.config = config
        self.cuda = cuda
        self.device = self.get_device()

        self.data = Data(self.config.data)

        self.model_class = self.get_model_class()
        self.model = self.model_class(
            vocab_size=self.data.vocab_size,
            **self.config.model.network.dict(),
        )  # type: AutoModel

        self.data.process(self.model)
        self.model.to(self.device)

    def get_device(self):
        if self.cuda == -1:
            return 'cpu'
        if self.cuda is None:
            return GPU.auto_choose(torch_format=True)
        return f'cuda:{self.cuda}'

    def get_model_class(self) -> Type[AutoModel]:
        model_dict = {'CBow': auto_model.CBow, 'SkipGram': auto_model.SkipGram}
        return model_dict[self.config.model.model]

    def train(self):
        dataset = self.data.sets['train']
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.config.model.policy.batch_size,
            shuffle=True,
            pin_memory=True,
        )

        for batch in tqdm(dataloader):
            labels = batch['oututs'].to(self.device)
            outputs = self.model(batch['inputs'].to(self.device))

