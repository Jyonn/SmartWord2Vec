import numpy as np


class LossRecorder:
    def __init__(self):
        self.losses = dict()

    def push(self, loss_dict: dict):
        for key, value in loss_dict.items():
            if key not in self.losses:
                self.losses[key] = []
            self.losses[key].append(value)

    def summarize(self):
        return {key: np.mean(value) for key, value in self.losses.items()}
