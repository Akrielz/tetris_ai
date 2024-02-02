from copy import deepcopy
from typing import Optional

import torch
from torch import nn

from tetris_ai.ai.rl.buffer import Buffer, TemporaryBuffer


class AgentPPO:
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
            episode_batch_size: int = 32,
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

        self.episode_batch_size = episode_batch_size

        # Move to device
        self._move_to_device()

        self.old_policy.eval()

    def _move_to_device(self):
        self.policy.to(self.device)
        self.old_policy.to(self.device)

    @torch.no_grad()
    def select_action(self, states: torch.Tensor) -> torch.Tensor:
        if states.device != self.device:
            states = states.to(self.device)

        action_output = self.old_policy.act(states)
        actions = action_output['actions']
        log_probs = action_output['log_probs']
        state_values = action_output['state_values']

        self.buffer.add('states', states)
        self.buffer.add('actions', actions)
        self.buffer.add('log_probs', log_probs)
        self.buffer.add('state_values', state_values)

        return actions

    @torch.no_grad()
    def _compute_discontinued_rewards(self, rewards: torch.Tensor, dones: torch.Tensor):

        len_episode, batch_size = rewards.shape

        discounted_rewards = torch.zeros(len_episode, batch_size, device=self.device)
        discounted_reward = torch.zeros(batch_size, device=self.device)
        for i, (reward, done) in enumerate(zip(reversed(rewards), reversed(dones))):
            # In case it is done, we reset the discounted reward to 0
            discounted_reward[done] = 0

            discounted_reward = reward + (self.gamma * discounted_reward)
            discounted_rewards[len_episode - i - 1] = discounted_reward

        # Normalize rewards according to its episode
        discounted_rewards = (discounted_rewards - discounted_rewards.mean(-2)) / (discounted_rewards.std(-2) + 1e-5)

        return discounted_rewards

    def update(self):
        buffer_states = self.buffer.states
        buffer_actions = self.buffer.actions
        buffer_log_probs = self.buffer.log_probs
        buffer_state_values = self.buffer.state_values
        buffer_rewards = self.buffer.rewards
        buffer_dones = self.buffer.dones

        episode_len = buffer_states.shape[0]

        # Compute discounted rewards
        discounted_rewards = self._compute_discontinued_rewards(buffer_rewards, buffer_dones)

        # Compute the advantages
        advantages = discounted_rewards.detach() - buffer_state_values.detach()

        # Optimize policy for K epochs
        for _ in range(self.num_epochs):
            for start in range(0, episode_len, self.episode_batch_size):
                end = start + self.episode_batch_size

                # Prepare the batches
                buffer_states_batch = buffer_states[start:end]
                buffer_actions_batch = buffer_actions[start:end]
                buffer_log_probs_batch = buffer_log_probs[start:end]
                discounted_rewards_batch = discounted_rewards[start:end]
                advantages_batch = advantages[start:end]

                # Evaluate old actions and values
                evaluation_output = self.policy.evaluate(buffer_states_batch, buffer_actions_batch)
                dist_entropy = evaluation_output['dist_entropy']
                log_probs = evaluation_output['log_probs']
                state_values = evaluation_output['state_values']

                # Compute the ratios
                ratios = torch.exp(log_probs - buffer_log_probs_batch)

                # Compute surrogate loss
                surrogate_1 = ratios * advantages_batch
                surrogate_2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages_batch
                loss = (
                    - torch.min(surrogate_1, surrogate_2)
                    + 0.5 * self.loss_fn(state_values, discounted_rewards_batch)
                    - 0.01 * dist_entropy
                )

                # Optimize the policy
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