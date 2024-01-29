from typing import List

from torch.nn import Linear
from vision_models_playground.components.convolutions import ResidualBlock
from vision_models_playground.models.classifiers import build_resnet_18, ResNet

from tetris_ai.ai.models.wrappers import critic_wrapper, actor_wrapper


def get_resnet_vmp(num_classes: int = 10):
    model = ResNet(
        num_classes=num_classes,
        in_channels=3,
        num_layers=[8, 8, 8],
        num_channels=[32, 64, 128],
        block=ResidualBlock
    )
    model.fc = Linear(128, num_classes)
    return model


def get_resnet_vmp_critic(state_dim: List[int]):
    resnet = get_resnet_vmp(num_classes=1)
    return critic_wrapper(resnet, state_dim, model_type='conv')


def get_resnet_vmp_actor(state_dim: List[int], action_dim: int):
    resnet = get_resnet_vmp(num_classes=action_dim)
    return actor_wrapper(resnet, state_dim, model_type='conv')
