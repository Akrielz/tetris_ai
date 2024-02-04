from copy import deepcopy
from typing import Dict, Tuple, Optional

import numpy as np
import torch

from tetris_ai.ai.env.torch_env import TorchEnv


class MaxDecisionTree:
    def __init__(self, env, max_depth: int = 1, num_workers: int = 0, device: torch.device = None):
        self.env = env
        self.max_depth = max_depth

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.device = device
        self.num_workers = num_workers

        self.board_width = env.board.width
        self.board_x_offset = 5

    def _determine_duplicates(self, boards: torch.Tensor):
        num_boards = len(boards)
        unique_states_mask = torch.ones(num_boards, dtype=torch.bool, device=self.device)
        sorted_indexes = torch.unique(boards, dim=0, return_inverse=True)[1]
        already_visited = set()

        for i in range(num_boards):
            if sorted_indexes[i].item() in already_visited:
                unique_states_mask[i] = False
                continue
            already_visited.add(sorted_indexes[i].item())

        return unique_states_mask

    def act_deterministic(
            self,
            depth: Optional[int] = None,
            reward: Optional[float] = 0.0,
            env: Optional[TorchEnv] = None
    ) -> Dict:
        # End node
        if depth == 0:
            return {
                'reward': reward,
                'action': None,
            }

        # Default arguments at the first call
        if depth is None:
            depth = self.max_depth

        if env is None:
            env = self.env

        # Perform all the actions in the current node
        actions = torch.arange(env.action_dim, device=self.device)
        env_after_actions = TorchEnv(env, batch_size=env.action_dim, num_workers=self.num_workers, device=self.device)
        outputs = env_after_actions.step(actions)
        rewards = outputs['rewards']

        # Determine which states have exactly the same boards
        boards = outputs['states'][:, :, self.board_x_offset:self.board_width + self.board_x_offset]
        unique_states_mask = self._determine_duplicates(boards)

        actions_final_rewards = []
        actions_final = []
        for i in range(len(actions)):
            if not unique_states_mask[i]:
                continue

            best_output = self.act_deterministic(
                depth - 1,
                reward + rewards[i],
                env_after_actions.envs[i],
            )
            best_reward = best_output['reward']
            actions_final_rewards.append(best_reward)
            actions_final.append(actions[i])

        action_final_rewards = torch.tensor(actions_final_rewards, device=self.device)

        best_index = torch.argmax(action_final_rewards)
        best_action = actions_final[best_index]
        best_reward = action_final_rewards[best_index]
        return {
            'reward': best_reward.item(),
            'action': best_action.item(),
        }
