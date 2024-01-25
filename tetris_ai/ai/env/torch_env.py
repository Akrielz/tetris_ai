from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from typing import Optional, List, Dict

import torch
from tensordict import TensorDict

from tetris_ai.ai.env.env import Env
from tetris_ai.game.tetris import TetrisEnv


class TorchEnv(Env):
    def __init__(
            self,
            env: TetrisEnv,
            batch_size: int = 1,
            num_workers: int = 0,
            device: Optional[torch.device] = None
    ):
        super().__init__()

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.envs = [deepcopy(env) for _ in range(batch_size)]
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device

    @property
    def action_dim(self):
        return self.envs[0].action_dim

    @property
    def state_dim(self):
        return self.envs[0].state_dim

    @staticmethod
    def _step_task(env: TetrisEnv, action: torch.LongTensor):
        action = action.item()
        return env.step(action)

    def _step_multiple_workers(self, actions: torch.Tensor) -> List[Dict]:
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [
                executor.submit(self._step_task, env, action)
                for env, action in zip(self.envs, actions)
            ]

            # Wait for all futures to complete
            results = [future.result() for future in futures]

        return results

    def _step_single_worker(self, actions: torch.Tensor) -> List[Dict]:
        results = [self._step_task(env, action) for env, action in zip(self.envs, actions)]
        return results

    def step(self, actions: torch.Tensor) -> TensorDict:
        results = self._step_multiple_workers(actions) if self.num_workers > 0\
            else self._step_single_worker(actions)

        # Unpack results
        results_dict = TensorDict({
                'states': torch.stack([torch.LongTensor(result["state"]) for result in results]),
                'rewards': torch.tensor([result["reward"] for result in results], dtype=torch.float),
                'dones': torch.tensor([result["done"] for result in results], dtype=torch.bool),
            },
            batch_size=self.batch_size
        )

        return results_dict

    def _reset_multiple_workers(self) -> List[Dict]:
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(env.reset) for env in self.envs]

            # Wait for all futures to complete
            results = [future.result() for future in futures]

        return results

    def _reset_single_worker(self) -> List[Dict]:
        results = [env.reset() for env in self.envs]
        return results

    def reset(self) -> TensorDict:
        results = self._reset_multiple_workers() if self.num_workers > 0\
            else self._reset_single_worker()

        # Unpack results
        results_dict = TensorDict({
                'states': torch.stack([torch.LongTensor(result['state']) for result in results]),
            },
            batch_size=self.batch_size
        )

        results_dict.to(self.device)

        return results_dict

    def rand_step(self) -> TensorDict:
        actions = torch.randint(0, self.envs[0].action_dim, (self.batch_size,))
        return self.step(actions)
