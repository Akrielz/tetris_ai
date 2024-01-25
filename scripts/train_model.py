import torch
from einops.layers.torch import Rearrange
from torch import nn

from tetris_ai.ai.actor_critic import ActorCritic
from tetris_ai.ai.buffer import TemporaryBuffer
from tetris_ai.ai.env.torch_env import TorchEnv
from tetris_ai.ai.env.transformed_env import TransformedEnv
from tetris_ai.ai.mlp import MultiLayerPerceptron
from tetris_ai.ai.ppo_agent import PPOAgent
from tetris_ai.ai.trainer import TrainerPPO
from tetris_ai.game.tetris import TetrisEnv


def main():
    # Prepare the environment
    env = TetrisEnv(height=23)
    env = TorchEnv(env, batch_size=1, num_workers=0)
    env = TransformedEnv(env)

    state_dim = env.state_dim
    action_dim = env.action_dim

    # Prepare the Actor & Critic
    actor = nn.Sequential(
        nn.Embedding(state_dim[-1], 3),
        Rearrange('b ... d -> b (... d)'),  # Flatten
        MultiLayerPerceptron([3 * state_dim[0] * state_dim[1], 128, 128, action_dim]),
        nn.Softmax(dim=-1)
    )

    critic = nn.Sequential(
        nn.Embedding(state_dim[-1], 3),
        Rearrange('b ... d -> b (... d)'),  # Flatten
        MultiLayerPerceptron([3 * state_dim[0] * state_dim[1], 128, 128, 1]),
    )

    actor_critic = ActorCritic(actor, critic)

    # Prepare the PPO Agent
    ppo_agent = PPOAgent(
        actor_critic,
        num_epochs=10,
        buffer=TemporaryBuffer(),
        device=torch.device('cuda'),
        lr_actor=1e-4,
        lr_critic=1e-4,
    )

    # Prepare Trainer
    trainer = TrainerPPO(
        env,
        ppo_agent,
        update_frequency=100,
        max_episode_length=1000,
    )

    # Train
    trainer.train(10000)


if __name__ == "__main__":
    main()