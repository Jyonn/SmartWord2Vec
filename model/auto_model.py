from torch import nn


class AutoModel(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            hidden_size: int,
            window_size: int,
            max_norm: float = None,
    ):
        super().__init__()

        self.embeddings = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=hidden_size,
            max_norm=max_norm,
        )
        self.linear = nn.Linear(
            in_features=hidden_size,
            out_features=vocab_size,
        )

        self.window_size = window_size

    def generate_data(self, sequence):
        raise NotImplementedError

    def forward(self, batch):
        raise NotImplementedError


class CBow(AutoModel):
    def generate_data(self, sequence):
        inputs, outputs = [], []
        for index in range(len(sequence) - 2 * self.window_size):
            window = sequence[index : index + self.window_size * 2 + 1]
            center = window.pop(self.window_size)
            inputs.append(window)
            outputs.append(center)
        return inputs, outputs

    def forward(self, batch):
        batch = self.embeddings(batch)
        batch = batch.mean(axis=1)
        batch = self.linear(batch)
        return batch


class SkipGram(AutoModel):
    def generate_data(self, sequence):
        inputs, outputs = [], []
        for index in range(len(sequence) - 2 * self.window_size):
            window = sequence[index : index + self.window_size * 2 + 1]
            center = window.pop(self.window_size)
            for neighbour in window:
                inputs.append(center)
                outputs.append(neighbour)
        return inputs, outputs

    def forward(self, batch):
        batch = self.embeddings(batch)
        batch = self.linear(batch)
        return batch
