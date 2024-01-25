import torch
from torch import nn
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(
            self,
            actor: nn.Module,
            critic: nn.Module,
    ):
        super().__init__()

        self.actor = actor
        self.critic = critic

    def act(self, state: torch.Tensor):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)

        action = dist.sample()
        action_log_probs = dist.log_prob(action)
        stave_value = self.critic(state)

        return action, action_log_probs, stave_value

    def evaluate(self, state: torch.Tensor, action: torch.Tensor):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)

        action_log_probs = dist.log_prob(action)
        state_value = self.critic(state)
        dist_entropy = dist.entropy()

        return action_log_probs, state_value, dist_entropy
