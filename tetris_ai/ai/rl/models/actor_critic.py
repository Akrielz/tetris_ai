from typing import Dict

import torch
from einops import rearrange
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

    def act(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        action_probs = self.actor(state)
        dist = Categorical(action_probs)

        action = dist.sample()
        action_log_probs = dist.log_prob(action)

        state_value = self.critic(state)

        return {
            'actions': action,
            'log_probs': action_log_probs,
            'state_values': state_value,
        }

    def act_deterministic(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        action_probs = self.actor(state)
        action = torch.argmax(action_probs, dim=-1)
        state_value = self.critic(state)

        return {
            'actions': action,
            'state_values': state_value,
        }

    def evaluate(self, state: torch.Tensor, action: torch.Tensor) -> Dict[str, torch.Tensor]:
        # State and Actions have both batch dimension and length dimension
        # Because of that, we will rearrange them behave as only a batch dimension both
        batch_dim = state.shape[0]
        state = rearrange(state, 'b l ... -> (b l) ...')

        action_probs = self.actor(state)
        action_probs = rearrange(action_probs, '(b l) ... -> b l ...', b=batch_dim)
        dist = Categorical(action_probs)

        action_log_probs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        state_value = self.critic(state)
        state_value = rearrange(state_value, '(b l) ... -> b l ...', b=batch_dim)

        return {
            'dist_entropy': dist_entropy,
            'log_probs': action_log_probs,
            'state_values': state_value,
        }
