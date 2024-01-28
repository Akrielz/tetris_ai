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


class ListToTensorBuffer(Buffer):

    def __init__(self):
        super().__init__()

        self._actions = []
        self._states = []
        self._log_probs = []
        self._rewards = []
        self._state_values = []
        self._dones = []

    def add_all(self, action: int, state: torch.Tensor, log_prob: float, reward: float, state_value: float, done: bool):
        self._actions.append(action)
        self._states.append(state)
        self._log_probs.append(log_prob)
        self._rewards.append(reward)
        self._state_values.append(state_value)
        self._dones.append(done)

    def add(self, key, value):
        internal_key = f'_{key}'
        getattr(self, internal_key).append(value)

    def clear(self):
        del self._actions[:]
        del self._states[:]
        del self._log_probs[:]
        del self._rewards[:]
        del self._state_values[:]
        del self._dones[:]

    def _tensor_view(self, key) -> torch.Tensor:
        internal_key = f'_{key}'
        return torch.stack(getattr(self, internal_key)).detach()

    @property
    def actions(self) -> torch.Tensor:
        return self._tensor_view('actions')

    @property
    def states(self) -> torch.Tensor:
        return self._tensor_view('states')

    @property
    def log_probs(self) -> torch.Tensor:
        return self._tensor_view('log_probs')

    @property
    def rewards(self) -> torch.Tensor:
        return self._tensor_view('rewards')

    @property
    def state_values(self) -> torch.Tensor:
        return self._tensor_view('state_values')

    @property
    def dones(self) -> torch.Tensor:
        return self._tensor_view('dones')


class TemporaryBuffer(ListToTensorBuffer):
    def __init__(self):
        super().__init__()


class RolloutBuffer(ListToTensorBuffer):
    def __init__(self, max_size: int):
        super().__init__()
        self.max_size = max_size

    def add_all(self, action: int, state: torch.Tensor, log_prob: float, reward: float, state_value: float, done: bool):
        if len(self._actions) >= self.max_size:
            self._actions.pop(0)
            self._states.pop(0)
            self._log_probs.pop(0)
            self._rewards.pop(0)
            self._state_values.pop(0)
            self._dones.pop(0)

        self._actions.append(action)
        self._states.append(state)
        self._log_probs.append(log_prob)
        self._rewards.append(reward)
        self._state_values.append(state_value)
        self._dones.append(done)

    def add(self, key, value):
        internal_key = f'_{key}'
        internal_list = getattr(self, internal_key)
        if len(internal_list) >= self.max_size:
            internal_list.pop(0)

        internal_list.append(value)

    def clear(self):
        # It doesn't need to remove anything
        pass