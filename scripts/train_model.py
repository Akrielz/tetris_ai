import torch
from einops.layers.torch import Rearrange
from torch import nn

from tetris_ai.ai.models.actor_critic import ActorCritic
from tetris_ai.ai.buffer import TemporaryBuffer
from tetris_ai.ai.env.torch_env import TorchEnv
from tetris_ai.ai.env.transformed_env import TransformedEnv
from tetris_ai.ai.models.mlp import MultiLayerPerceptron
from tetris_ai.ai.agent_ppo import AgentPPO
from tetris_ai.ai.trainer import TrainerPPO
from tetris_ai.game.cell import Cell
from tetris_ai.game.tetris import TetrisEnv


def main():
    # Prepare general info
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Prepare the environment
    env = TetrisEnv(height=23, sparse_rewards=True, action_penalty=False)
    env = TorchEnv(env, batch_size=1, num_workers=0, device=device)
    env = TransformedEnv(env)

    state_dim = env.state_dim
    action_dim = env.action_dim

    # Prepare the Actor & Critic
    actor = nn.Sequential(
        nn.Embedding(state_dim[-1], 3, padding_idx=Cell.PAD.value),
        Rearrange('... w h d -> ... (w h d)'),  # Flatten
        MultiLayerPerceptron([3 * state_dim[0] * state_dim[1], 128, 64, 32, 16, action_dim]),
        nn.Softmax(dim=-1)
    )

    critic = nn.Sequential(
        nn.Embedding(state_dim[-1], 3, padding_idx=Cell.PAD.value),
        Rearrange('... w h d -> ... (w h d)'),  # Flatten
        MultiLayerPerceptron([3 * state_dim[0] * state_dim[1], 128, 64, 32, 16, 1]),
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