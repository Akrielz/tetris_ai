from abc import ABC

import numpy as np
import torch


class Buffer(ABC):
    def add_all(self, action: int, state: np.ndarray, log_prob: float, reward: float, state_value: float, done: bool):
        raise NotImplementedError

    def add(self, key, value):
        raise NotImplementedError

    def clear(self):
        raise NotImplementedError


class RolloutBuffer(Buffer):
    def __init__(self, buffer_size: int):
        super().__init__(buffer_size)

        self.actions = []
        self.states = []
        self.log_probs = []
        self.rewards = []
        self.state_values = []
        self.dones = []

    def add_all(self, action: int, state: torch.Tensor, log_prob: float, reward: float, state_value: float, done: bool):
        self.actions.append(action)
        self.states.append(state)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.state_values.append(state_value)
        self.dones.append(done)

    def add(self, key, value):
        getattr(self, key).append(value)

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.log_probs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.dones[:]