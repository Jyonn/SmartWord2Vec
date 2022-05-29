import torch
from torch.utils.data import Dataset

from utils.splitter import Splitter


class Word2VecDataset(Dataset):
    def __init__(
            self,
            inputs: list,
            outputs: list,
            mode: str,
            splitter: Splitter,
    ):
        super(Word2VecDataset, self).__init__()

        self.inputs = inputs
        self.outputs = outputs
        self.mode = mode
        self.splitter = splitter

        assert len(self.inputs) == len(self.outputs)

        self.split_range = splitter.divide(len(self.inputs))[self.mode]

    def pack_sample(self, index):
        inputs = self.inputs[index]
        outputs = self.outputs[index]
        return dict(
            inputs=torch.tensor(inputs, dtype=torch.long),
            outputs=torch.tensor(outputs, dtype=torch.long),
        )

    def __getitem__(self, index):
        index += self.split_range[0]
        return self.pack_sample(index)

    def __len__(self):
        return self.split_range[1] - self.split_range[0]
