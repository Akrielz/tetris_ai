import gym
import torch
from einops.layers.torch import Rearrange
from torch.nn import Sequential, Softmax

from tetris_ai.ai.rl.agent_ppo import AgentPPO
from tetris_ai.ai.rl.buffer import TemporaryBuffer
from tetris_ai.ai.env.torch_env import TorchEnv
from tetris_ai.ai.env.transformed_env import TransformedEnv
from tetris_ai.ai.rl.models.actor_critic import ActorCritic
from tetris_ai.ai.rl.models.mlp import MultiLayerPerceptron
from tetris_ai.ai.rl.trainer import TrainerPPO


class CartPoleEnv:
    def __init__(self: str):
        self.env = gym.make('CartPole-v1')

    def reset(self):
        state, _ = self.env.reset()
        return {
            'state': state,
        }

    def step(self, action):
        state, reward, done, _, _ = self.env.step(action)
        return {
            'state': state,
            'reward': reward,
            'done': done,
        }


def main():
    """
    This code is to make sure the PPO algorithm works.
    """

    # Prepare general info
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Prepare the environment
    env = CartPoleEnv()
    env = TorchEnv(env, batch_size=1, num_workers=0, device=device, state_type='float')
    env = TransformedEnv(env)

    # Prepare the Actor & Critic
    state_dim = 4
    action_dim = 2

    actor = Sequential(
        MultiLayerPerceptron(inner_dims=[state_dim, 64, 64, action_dim], activation='tanh'),
        Softmax(dim=-1),
    )
    critic = Sequential(
        MultiLayerPerceptron(inner_dims=[state_dim, 64, 64, 1], activation='tanh'),
        Rearrange('... 1 -> ...'),
    )

    actor_critic = ActorCritic(actor, critic)

    # Prepare the PPO Agent
    ppo_agent = AgentPPO(
        actor_critic,
        num_epochs=10,
        buffer=TemporaryBuffer(),
        device=device,
        lr_actor=3e-4,
        lr_critic=1e-3,
        gamma=0.99,
        episode_batch_size=100,
    )

    # Prepare Trainer
    trainer = TrainerPPO(
        env,
        ppo_agent,
        update_frequency=400,
        episode_length_start=100,
        episode_length_increase=0.2,
    )

    # Train
    trainer.train(int(1e7))


if __name__ == "__main__":
    main()