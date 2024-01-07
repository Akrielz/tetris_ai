from abc import ABC
from typing import List

import torch
from torch import nn
from torch.distributions import Categorical


class ActorCritic(nn.Module, ABC):
    def act(self, state: torch.Tensor):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)

        action = dist.sample()
        log_prob = dist.log_prob(action)
        stave_value = self.critic(state)

        return action, log_prob, stave_value

    def evaluate(self, state: torch.Tensor, action: torch.Tensor):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)

        action_log_probs = dist.log_prob(action)
        state_value = self.critic(state)
        dist_entropy = dist.entropy()

        return action_log_probs, state_value, dist_entropy


class LinearActorCritic(ActorCritic):
    def __init__(
            self,
            state_dim: int,
            inner_dims: List[int],
            output_dim: int,
    ):
        super().__init__()

        if inner_dims is None:
            inner_dims = [128, 128]

        embedding = nn.Embedding(state_dim, inner_dims[0])

        layers = []
        for i in range(len(inner_dims) - 1):
            layers.append(nn.Linear(inner_dims[i], inner_dims[i + 1]))
            layers.append(nn.ReLU())

        self.actor = nn.Sequential(
            embedding, *layers, nn.Linear(inner_dims[-1], output_dim), nn.Softmax(dim=1)
        )
        self.critic = nn.Sequential(
            embedding, *layers, nn.Linear(inner_dims[-1], 1)
        )
