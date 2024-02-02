from asyncio import sleep

import torch

from tetris_ai.ai.decision_tree.max_tree import MaxDecisionTree
from tetris_ai.game.tetris import MultiActionTetrisEnv
from tetris_ai.game.visualizer import VisualTetrisEnv


def main():
    # Prepare general info
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')

    # Prepare the environment
    env = MultiActionTetrisEnv(
        height=23, sparse_rewards=True, action_penalty=False, force_down_every_n_moves=0,
    )
    visualizer = VisualTetrisEnv(env, block_dim=20)
    decision_tree = MaxDecisionTree(env, max_depth=1, num_workers=4, device=device)

    # Test the model
    env.reset()
    cumulative_reward = 0.0
    while True:
        visualizer.render()
        print(f"Cumulative Reward: {cumulative_reward}")

        action = decision_tree.act_deterministic()['action']
        output = env.step(action)
        done = output['done']
        reward = output['reward']
        cumulative_reward += reward

        if done:
            env.reset()
            cumulative_reward = 0.0

        # sleep(0.4)


if __name__ == "__main__":
    main()