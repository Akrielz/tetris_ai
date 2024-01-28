from typing import List

import torch
from torch import nn
from torch.nn import Dropout

from tetris_ai.ai.models.wrappers import actor_wrapper, critic_wrapper


class MultiLayerPerceptron(torch.nn.Module):
    def __init__(
            self,
            inner_dims: List[int],
            dropout: float = 0.1,
            activation: str='relu',
    ):
        super().__init__()

        if activation == 'relu':
            activation_fn = nn.ReLU()
        elif activation == 'tanh':
            activation_fn = nn.Tanh()
        else:
            raise ValueError(f'Unknown activation function: {activation}')

        if inner_dims is None:
            inner_dims = [128, 128]

        self.layers = nn.ModuleList()
        for i in range(len(inner_dims) - 2):
            self.layers.append(nn.Linear(inner_dims[i], inner_dims[i + 1]))
            self.layers.append(Dropout(dropout))
            self.layers.append(activation_fn)

        self.layers.append(nn.Linear(inner_dims[-2], inner_dims[-1]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)

        return x


def get_mlp_critic(state_dim: List[int]):
    mlp = MultiLayerPerceptron(inner_dims=[state_dim[0] * state_dim[1] * 3, 16, 32, 32, 32, 32, 32, 1])
    return critic_wrapper(mlp, state_dim, model_type='mlp')


def get_mlp_actor(state_dim: List[int], action_dim: int):
    mlp = MultiLayerPerceptron(inner_dims=[state_dim[0] * state_dim[1] * 3, 16, 32, 32, 32, 32, 32, action_dim])
    return actor_wrapper(mlp, state_dim, model_type='mlp')