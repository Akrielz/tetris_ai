import torch
from torchmetrics import Metric


class RewardTracker(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("rewards", default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('num_episodes', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, rewards: torch.Tensor):
        self.rewards += torch.mean(rewards, dim=0)
        self.num_episodes += rewards.shape[0]

    def compute(self):
        return self.rewards / self.num_episodes
