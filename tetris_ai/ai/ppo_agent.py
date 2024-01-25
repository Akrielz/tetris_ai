from copy import deepcopy
from typing import Optional

import torch
from torch import nn

from tetris_ai.ai.buffer import Buffer, TemporaryBuffer


class PPOAgent:
    def __init__(
            self,
            actor_critic: nn.Module,
            num_epochs: int,
            eps_clip: float = 0.01,  # Clip parameter for PPO
            gamma: float = 0.99,  # Reduce factor
            buffer: Optional[Buffer] = None,
            device: Optional[torch.device] = None,
            lr_actor: float = 0.001,
            lr_critic: float = 0.001,
    ):
        super().__init__()

        # Init default params
        if buffer is None:
            buffer = TemporaryBuffer()

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        optimizer = torch.optim.Adam([
            {'params': actor_critic.actor.parameters(), 'lr': lr_actor},
            {'params': actor_critic.critic.parameters(), 'lr': lr_critic}
        ])

        loss_fn = nn.MSELoss()

        # Save params
        self.policy = actor_critic
        self.old_policy = deepcopy(actor_critic)

        self.optimizer = optimizer

        self.loss_fn = loss_fn

        self.num_epochs = num_epochs
        self.eps_clip = eps_clip
        self.gamma = gamma

        self.buffer = buffer
        self.device = device

        # Move to device
        self._move_to_device()

    def _move_to_device(self):
        self.policy.to(self.device)
        self.old_policy.to(self.device)

    @torch.no_grad()
    def select_action(self, states: torch.Tensor):
        state = torch.LongTensor(states).to(self.device)
        action, log_prob, state_value = self.policy.act(state)

        self.buffer.add('states', state)
        self.buffer.add('actions', action)
        self.buffer.add('log_probs', log_prob)
        self.buffer.add('state_values', state_value)

        return action

    def update(self):
        rewards = torch.zeros(len(self.buffer.rewards), device=self.device)
        discounted_reward = 0
        for i, (reward, done) in enumerate(zip(reversed(self.buffer.rewards), reversed(self.buffer.done))):
            if done:
                discounted_reward = 0

            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards[len(self.buffer.rewards) - i - 1] = discounted_reward

        # Normalize rewards
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # Convert to tensors
        old_states = torch.stack(self.buffer.states).to(self.device).detach()
        old_actions = torch.stack(self.buffer.actions).to(self.device).detach()
        old_log_probs = torch.stack(self.buffer.log_probs).to(self.device).detach()
        old_state_values = torch.stack(self.buffer.state_values).to(self.device).detach()
        old_rewards = rewards.to(self.device).detach()

        # Compute the advantages
        advantages = rewards - old_state_values

        # Optimize policy for K epochs
        for _ in range(self.num_epochs):
            # Evaluate old actions and values
            log_probs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # Compute the ratios
            ratios = torch.exp(log_probs - old_log_probs.detach())

            # Compute surrogate loss
            surrogate_1 = ratios * advantages.detach()
            surrogate_2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages.detach()
            loss = -torch.min(surrogate_1, surrogate_2) + 0.5 * self.loss_fn(state_values, old_rewards) - 0.01 * dist_entropy

            # Take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy
        self.old_policy.load_state_dict(self.policy.state_dict())

        # Clear buffer
        self.buffer.clear()

    def save(self, path: str):
        torch.save(self.policy.state_dict(), path)

    def load(self, path: str):
        self.policy.load_state_dict(torch.load(path))
        self.old_policy.load_state_dict(torch.load(path))