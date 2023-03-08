import warnings

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class CascadeCodebookCluster(nn.Module):
    def __init__(
            self,
            embed_dim,
            vocab_size,
            num_layers,
            commitment_cost=0.25,
            individual=True,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.commitment_cost = commitment_cost
        self.individual = individual

        assert vocab_size > 4, "vocab_size must be greater than 1"
        assert num_layers >= 1, "num_layers must be greater than 0"

        if 2 ** num_layers > vocab_size:
            warnings.warn(f"num_layers is too large, set to {self.num_layers}")
            self.num_layers = int(np.log2(vocab_size))

        self.top_cluster_size = int(vocab_size ** (1.0 / (num_layers + 1)) + 0.5)
        self.cluster_size = [self.top_cluster_size]
        for i in range(num_layers - 1):
            self.cluster_size.append(self.top_cluster_size * self.cluster_size[i])
        self.cluster_size = self.cluster_size[::-1]

        self.codebooks = nn.ModuleList()
        for i in range(num_layers):
            self.codebooks.append(nn.Embedding(self.cluster_size[i], embed_dim))

    def quantize(self, embeds):  # [B, ..., D]
        shape = embeds.shape
        embeds = embeds.view(-1, self.embed_dim)  # [B * ..., D]
        qembeds = []
        for i in range(self.num_layers):
            distances = torch.sum(embeds ** 2, dim=1, keepdim=True) + \
                        torch.sum(self.codebooks[i].weight ** 2, dim=1) - \
                        2 * torch.matmul(embeds, self.codebooks[i].weight.t())
            indices = torch.argmin(distances, dim=-1).unsqueeze(1)
            placeholder = torch.zeros(indices.shape[0], self.cluster_size[i], device=embeds.device)
            placeholder.scatter_(1, indices, 1)
            inner_embeds = torch.matmul(placeholder, self.codebooks[i].weight).view(embeds.shape)
            qembeds.append(inner_embeds.view(shape))
            if not self.individual:
                embeds = inner_embeds
        return qembeds

    def get_qloss(self, embeds, qembeds):
        compare_embeds = embeds
        # qembeds = [embeds, *qembeds]
        q_loss = 0
        for i in range(self.num_layers):
            q_loss += F.mse_loss(qembeds[i].detach(), compare_embeds) * self.commitment_cost \
                      + F.mse_loss(qembeds[i], compare_embeds.detach())
            if not self.individual:
                compare_embeds = qembeds[i]
        return q_loss
