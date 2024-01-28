from typing import List, Literal

from einops.layers.torch import Rearrange
from torch import nn

from tetris_ai.game.cell import Cell


def _compute_input_rearrange(type: Literal['mlp', 'conv']) -> nn.Module:
    if type == 'mlp':
        return Rearrange('... h w d -> ... (h w d)')
    elif type == 'conv':
        return Rearrange('... h w d -> ... d h w')
    else:
        raise ValueError(f'Unknown type {type}')


def critic_wrapper(model: nn.Module, state_dim: List[int], model_type: Literal['mlp', 'conv'] = 'mlp'):
    input_rearranger = _compute_input_rearrange(model_type)

    return nn.Sequential(
        nn.Embedding(state_dim[-1], 3, padding_idx=Cell.PAD.value),
        input_rearranger,
        model,
        Rearrange('... 1 -> ...'),
    )


def actor_wrapper(model: nn.Module, state_dim: List[int], model_type: Literal['mlp', 'conv'] = 'mlp'):
    input_rearranger = _compute_input_rearrange(model_type)
    return nn.Sequential(
        nn.Embedding(state_dim[-1], 3, padding_idx=Cell.PAD.value),
        input_rearranger,
        model,
        nn.Softmax(dim=-1)
    )
