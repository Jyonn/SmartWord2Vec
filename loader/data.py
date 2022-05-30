from UniTok import UniDep

from loader.dataset import Word2VecDataset
from model.auto_model import AutoModel
from utils.smart_printer import printer
from utils.splitter import Splitter


class Data:
    def __init__(self, data_config):
        self.config = data_config
        self.sequence_col = self.config.data.sequence_col

        self.print = printer.DATA_Cr_

        self.depot = UniDep(self.config.data.dir)
        self.depot.shuffle(shuffle=self.config.data.shuffle)

        self.splitter = Splitter()
        for mode, mode_config in self.config.data.modes:
            self.splitter.add(mode, mode_config.weight)

        self.vocab_size = self.depot.get_vocab_size(self.sequence_col)
        self.sets = dict()
        self.loaders = dict()

    def process(self, model: AutoModel):
        inputs, outputs = [], []
        for sample in self.depot:
            sequence = sample[self.sequence_col]
            inputs_, outputs_ = model.generate_data(sequence)
            inputs.extend(inputs_)
            outputs.extend(outputs_)

        for mode, _ in self.config.data.modes:
            self.sets[mode] = Word2VecDataset(
                inputs=inputs,
                outputs=outputs,
                mode=mode,
                splitter=self.splitter,
            )
            self.print(f'{mode} dataset created, with {len(self.sets[mode])} samples in total')

