from typing import List

import torch

from tetris_ai.ai.rl.agent_ppo import AgentPPO
from tetris_ai.ai.rl.buffer import RolloutBuffer
from tetris_ai.ai.env.torch_env import TorchEnv
from tetris_ai.ai.env.transformed_env import TransformedEnv
from tetris_ai.ai.rl.models.actor_critic import ActorCritic
from tetris_ai.ai.rl.models.resnet import get_resnet_vmp_actor, get_resnet_vmp_critic
from tetris_ai.ai.rl.trainer import TrainerPPO
from tetris_ai.game.tetris import MultiActionTetrisEnv


def load_model(
        model_path: str,
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

    # Prepare the Actor & Critic
    # actor = get_resnet_vmp_actor(state_dim, action_dim)
    # critic = get_resnet_vmp_critic(state_dim)
    #
    # actor_critic = ActorCritic(actor, critic)
    model_path = 'models/train/resnet_vmp_rollout_buffer/2024-01-30_07-09-27/best.pt'
    actor_critic = load_model(model_path, state_dim, action_dim)

    # Prepare the PPO Agent
    ppo_agent = AgentPPO(
        actor_critic,
        num_epochs=2,
        buffer=RolloutBuffer(30000),
        device=device,
        lr_actor=3e-4,
        lr_critic=1e-3,
        gamma=0.99,
        episode_batch_size=5000,
    )

    # Prepare Trainer
    trainer = TrainerPPO(
        env,
        ppo_agent,
        update_frequency=400,
        episode_length_start=100,
        episode_length_increase=0.2,
        model_name='resnet_vmp_rollout_buffer',
    )

    # Train
    trainer.train(int(1e7))


if __name__ == "__main__":
    main()