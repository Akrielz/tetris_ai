from abc import ABC

import torch
from tensordict import TensorDict


class Env(ABC):
    def __init__(self):
        super().__init__()

    @property
    def action_dim(self):
        raise NotImplementedError

    @property
    def state_dim(self):
        raise NotImplementedError

    def step(self, actions: torch.Tensor) -> TensorDict:
        raise NotImplementedError

    def reset(self) -> TensorDict:
        raise NotImplementedError

    def rand_step(self) -> TensorDict:
        raise NotImplementedError
