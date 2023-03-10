import warnings
from typing import List, Union, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers.activations import ACT2FN


class TransformLayer(nn.Module):
    """
    Transform layer for Classifier
    """
    def __init__(
            self,
            embed_dim,
            activation_function,
            layer_norm_eps=None,
    ):
        super(TransformLayer, self).__init__()
        self.transform = nn.Linear(embed_dim, embed_dim)
        self.transform_act_fn = ACT2FN[activation_function]
        self.layer_norm = nn.LayerNorm(embed_dim, eps=layer_norm_eps or 1e-5)

    def forward(self, hidden_states) -> torch.Tensor:
        # hidden_states = self.transform(hidden_states)
        # hidden_states = self.transform_act_fn(hidden_states)
        # hidden_states = self.layer_norm(hidden_states)
        return hidden_states


class DecoderLayer(nn.Module):
    """
    Decoder layer for Classifier, projecting hidden states to vocab_size
    """
    def __init__(
            self,
            embed_dim,
            vocab_size,
    ):
        super(DecoderLayer, self).__init__()
        self.decoder = nn.Linear(embed_dim, vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(vocab_size), requires_grad=True)
        self.decoder.bias = self.bias

    def forward(self, hidden_states) -> torch.Tensor:
        return self.decoder(hidden_states)


class CascadeCodebookCluster(nn.Module):
    """
    Cascade Codebook Cluster Classifier
    """
    def __init__(
            self,
            embed_dim,
            vocab_size,
            num_layers,
            commitment_cost=0.25,  # vector quantization loss
            layer_connect=False,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.commitment_cost = commitment_cost
        self.layer_connect = layer_connect

        assert vocab_size >= 1, "vocab_size must be greater than 1"
        assert num_layers >= 0 and isinstance(num_layers, int), "num_layers must be a non-negative integer"

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

        # general classifier
        self.transform_layer = TransformLayer(
            embed_dim=self.embed_dim,
            activation_function='relu',
        )

        self.decoder_layer = DecoderLayer(
            embed_dim=self.embed_dim,
            vocab_size=self.vocab_size,
        )

    def transform(self, embeds):
        return self.transform_layer(embeds)
        # return embeds

    def quantize(
            self,
            embeds,
            with_loss=False,
            transformed=False,
    ) -> Union[
        List[torch.Tensor],
        Tuple[List[torch.Tensor], torch.Tensor]
    ]:
        if not transformed:
            embeds = self.transform(embeds)
        compare_embeds = embeds  # for loss calculation

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
            if self.layer_connect:
                embeds = inner_embeds
        if not with_loss:
            return qembeds

        q_loss = torch.tensor(0, dtype=torch.float, device=embeds.device)
        for i in range(self.num_layers):
            q_loss += F.mse_loss(qembeds[i].detach(), compare_embeds) * self.commitment_cost \
                      + F.mse_loss(qembeds[i], compare_embeds.detach())
            if self.layer_connect:
                compare_embeds = qembeds[i]
        return qembeds, q_loss

    # def get_qloss(self, embeds, qembeds):
    #     compare_embeds = embeds
    #     q_loss = 0
    #     for i in range(self.num_layers):
    #         q_loss += F.mse_loss(qembeds[i].detach(), compare_embeds) * self.commitment_cost \
    #                   + F.mse_loss(qembeds[i], compare_embeds.detach())
    #         if self.layer_connect:
    #             compare_embeds = qembeds[i]
    #     return q_loss

    def classify(
            self,
            embeds,
            transformed=False,
    ):
        if not transformed:
            embeds = self.transform(embeds)
        return self.decoder_layer(embeds)
