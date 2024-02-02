from copy import deepcopy
from typing import Dict, Tuple, Optional

import torch

from tetris_ai.ai.env.torch_env import TorchEnv


class MaxDecisionTree:
    def __init__(self, env, max_depth: int = 3, num_workers: int = 0, device: torch.device = None):
        self.env = env
        self.max_depth = max_depth

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.device = device
        self.num_workers = num_workers

    def act_deterministic(
            self,
            depth: Optional[int] = None,
            reward: Optional[float] = 0.0,
            env: Optional[TorchEnv] = None
    ) -> Dict:
        if depth is None:
            depth = self.max_depth

        if depth == 0:
            return {
                'reward': reward,
                'action': None,
            }

        if env is None:
            env = self.env

        actions = torch.arange(env.action_dim, device=self.device)
        env_after_actions = TorchEnv(env, batch_size=env.action_dim)
        rewards = env_after_actions.step(actions)['rewards']

        action_final_rewards = []
        for i in range(len(actions)):
            best_output = self.act_deterministic(
                depth - 1,
                reward + rewards[i],
                env_after_actions.envs[i],
            )
            best_reward = best_output['reward']
            action_final_rewards.append(best_reward)

        action_final_rewards = torch.tensor(action_final_rewards, device=self.device)
        best_action = torch.argmax(action_final_rewards)
        best_reward = action_final_rewards[best_action]
        return {
            'reward': best_reward.item(),
            'action': best_action.item(),
        }
