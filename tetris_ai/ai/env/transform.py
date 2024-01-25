from abc import ABC
from typing import List, Optional

from tensordict import TensorDict
from torch import nn


class Transform(ABC, nn.Module):
    def __init__(
            self,
            in_keys: Optional[List[str]] = None,
            out_keys: Optional[List[str]] = None,
    ):
        super().__init__()

        self.in_keys = in_keys
        if out_keys is None:
            out_keys = in_keys
        self.out_keys = out_keys

    def forward(
            self,
            tensor_dict: TensorDict
    ) -> TensorDict:
        raise NotImplementedError()


class Identity(Transform):
    def __init__(self,):
        super().__init__()

    def forward(
            self,
            tensor_dict: TensorDict
    ) -> TensorDict:
        return tensor_dict
