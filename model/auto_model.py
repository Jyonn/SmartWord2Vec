# import torch
import torch
from torch import nn

# from torch.nn import functional as F

from model.c3 import CascadeCodebookCluster


class AutoModel(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            hidden_size: int,
            window_size: int,
            codebook_layers: int = 1,
            commitment_cost: float = 0.25,
            max_norm: float = None,
    ):
        super().__init__()

        self.embeddings = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=hidden_size,
            max_norm=max_norm,
        )
        # self.linear = nn.Linear(
        #     in_features=hidden_size,
        #     out_features=vocab_size,
        # )

        self.window_size = window_size
        # self.cookbook_size = cookbook_size
        self.codebook_layers = codebook_layers
        self.commitment_cost = commitment_cost

        # if self.cookbook_size:
        #     self.cookbook = nn.Embedding(
        #         num_embeddings=cookbook_size,
        #         embedding_dim=hidden_size,
        #     )
        #     self.cookbook.weight.data.normal_(0, 0.1)
        self.cascade_codebook = CascadeCodebookCluster(
            embed_dim=hidden_size,
            vocab_size=vocab_size,
            num_layers=self.codebook_layers,
            commitment_cost=commitment_cost,
        )

        # cookbook initialization

    # def quantize(self, embeds: torch.Tensor):
    #     distances = torch.sum(embeds ** 2, dim=1, keepdim=True) + \
    #                 torch.sum(self.cookbook.weight ** 2, dim=1) - \
    #                 2 * torch.matmul(embeds, self.cookbook.weight.t())  # type: torch.Tensor
    #     encoding_indices = torch.argmin(distances, dim=-1).unsqueeze(1)  # type: torch.Tensor
    #     encodings = torch.zeros(encoding_indices.shape[0], self.cookbook_size, device=embeds.device)
    #     encodings.scatter_(1, encoding_indices, 1)
    #     quantized = torch.matmul(encodings, self.cookbook.weight).view(embeds.shape)
    #     return quantized
    #
    # def get_quantized_loss(self, embeds):
    #     quantized = self.quantize(embeds)
    #     quantized_loss = F.mse_loss(quantized.detach(), embeds) * self.commitment_cost \
    #                      + F.mse_loss(quantized, embeds.detach())
    #     return quantized, quantized_loss

    def internal_forward(self, embeds):
        quantized, qloss = self.cascade_codebook.quantize(embeds, with_loss=True, transformed=False)
        distribution = self.cascade_codebook.classify(embeds, transformed=False)
        return distribution, qloss

    def generate_data(self, sequence):
        raise NotImplementedError

    def forward(self, batch):
        raise NotImplementedError


class CBow(AutoModel):
    def generate_data(self, sequence):
        inputs, outputs = [], []
        for index in range(len(sequence) - 2 * self.window_size):
            window = sequence[index: index + self.window_size * 2 + 1]
            center = window.pop(self.window_size)
            inputs.append(window)
            outputs.append(center)
        return inputs, outputs

    def forward(self, batch):
        batch = self.embeddings(batch)
        batch = batch.mean(axis=1)
        return self.internal_forward(batch)
        # return self.linear(batch), torch.tensor(0)


class SkipGram(AutoModel):
    def generate_data(self, sequence):
        inputs, outputs = [], []
        for index in range(len(sequence) - 2 * self.window_size):
            window = sequence[index: index + self.window_size * 2 + 1]
            center = window.pop(self.window_size)
            for neighbour in window:
                inputs.append(center)
                outputs.append(neighbour)
        return inputs, outputs

    def forward(self, batch):
        batch = self.embeddings(batch)
        return self.internal_forward(batch)
