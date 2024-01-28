import torch

from tetris_ai.ai.models.actor_critic import ActorCritic
from tetris_ai.ai.buffer import TemporaryBuffer
from tetris_ai.ai.env.torch_env import TorchEnv
from tetris_ai.ai.env.transformed_env import TransformedEnv
from tetris_ai.ai.agent_ppo import AgentPPO
from tetris_ai.ai.models.conv_vision_trasnformer import get_conv_vision_transformer_actor, get_conv_vision_transformer_critic
from tetris_ai.ai.models.resnet import get_resnet_vmp_critic, get_resnet_vmp_actor
from tetris_ai.ai.trainer import TrainerPPO
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
    actor = get_resnet_vmp_actor(state_dim, action_dim)
    critic = get_resnet_vmp_critic(state_dim)

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