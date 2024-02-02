from typing import List

import torch
from vision_models_playground.models.classifiers import ConvVisionTransformer

from tetris_ai.ai.rl.models.wrappers import critic_wrapper, actor_wrapper


def get_conv_vision_transformer(num_classes: int = 10):
    return ConvVisionTransformer(
        in_channels=3,
        num_classes=num_classes,
        patch_size=[3, 3, 3],
        patch_stride=[2, 2, 2],
        patch_padding=[1, 1, 1],
        embedding_dim=[8, 8, 8],
        depth=[2, 2, 4],
        num_heads=[1, 3, 6],
        ff_hidden_dim=[16, 16, 16],
        qkv_bias=[True, True, True],
        drop_rate=[0.1, 0.1, 0.1],
        attn_drop_rate=[0.1, 0.1, 0.1],
        drop_path_rate=[0.0, 0.0, 0.1],
        kernel_size=[3, 3, 3],
        stride_kv=[2, 2, 2],
        stride_q=[1, 1, 1],
        padding_kv=[1, 1, 1],
        padding_q=[1, 1, 1],
        method=['conv', 'conv', 'conv'],
    )


def get_conv_vision_transformer_critic(state_dim: List[int]):
    resnet = get_conv_vision_transformer(num_classes=1)
    return critic_wrapper(resnet, state_dim, model_type='conv')


def get_conv_vision_transformer_actor(state_dim: List[int], action_dim: int):
    resnet = get_conv_vision_transformer(num_classes=action_dim)
    return actor_wrapper(resnet, state_dim, model_type='conv')
