from time import sleep
from typing import List

import torch

from tetris_ai.ai.env.torch_env import TorchEnv
from tetris_ai.ai.env.transformed_env import TransformedEnv
from tetris_ai.ai.models.actor_critic import ActorCritic
from tetris_ai.ai.models.resnet import get_resnet_vmp_actor, get_resnet_vmp_critic
from tetris_ai.game.actions import LimitedAction
from tetris_ai.game.tetris import TetrisEnv, MultiActionTetrisEnv


def load_model(
        model_path: str,
        device: torch.device,
        state_dim: List[int],
        action_dim: int
):
    # Prepare the Actor & Critic
    actor = get_resnet_vmp_actor(state_dim, action_dim)
    critic = get_resnet_vmp_critic(state_dim)

    actor_critic = ActorCritic(actor, critic)

    # Load the model
    state_dict = torch.load(model_path)
    actor_critic.load_state_dict(state_dict)

    # Set to eval
    actor_critic.eval()
    actor_critic.to(device)

    return actor_critic


def main():
    # Prepare general info
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Prepare the environment
    env = MultiActionTetrisEnv(
        height=23, sparse_rewards=True, action_penalty=False, force_down_every_n_moves=0,
    )
    env = TorchEnv(env, batch_size=1, num_workers=0, device=device)
    env = TransformedEnv(env)

    state_dim = env.state_dim
    action_dim = env.action_dim

    # Load the model
    model_path = 'models/train/resnet_vmp_rollout_buffer/2024-01-30_07-09-27/best.pt'
    actor_critic = load_model(model_path, device, state_dim, action_dim)

    # Test the model
    state = env.reset()['states']

    while True:
        env.env.envs[0].render_console()

        action = actor_critic.act(state)['actions']

        output = env.step(action)
        state = output['states']
        done = output['dones']

        if torch.any(done):
            env.reset()

        sleep(0.4)


if __name__ == "__main__":
    main()