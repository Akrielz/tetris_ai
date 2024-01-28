from typing import List

import torch
from torch import nn
from torch.nn import Dropout


class MultiLayerPerceptron(torch.nn.Module):
    def __init__(
            self,
            inner_dims: List[int],
            dropout: float = 0.1,
    ):
        super().__init__()

        if inner_dims is None:
            inner_dims = [128, 128]

        self.layers = nn.ModuleList()
        for i in range(len(inner_dims) - 2):
            self.layers.append(nn.Linear(inner_dims[i], inner_dims[i + 1]))
            self.layers.append(Dropout(dropout))
            self.layers.append(nn.ReLU())

        self.layers.append(nn.Linear(inner_dims[-2], inner_dims[-1]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)

        return x
