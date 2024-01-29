import os
from datetime import datetime
from typing import Optional

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from tetris_ai.ai.env.env import Env
from tetris_ai.ai.agent_ppo import AgentPPO
from tetris_ai.ai.reward_tracker import RewardTracker


class TrainerPPO:
    def __init__(
            self,
            env: Env,
            agent: AgentPPO,
            update_frequency: int = 100,
            episode_length_start: float = 100,
            episode_length_increase: float = 0,
            save_dir: Optional[str] = None,
            model_name: str = "ppo_model",
            save_every_n_episodes: int = 100,
            lr_scheduler_params: Optional[dict] = None,
    ):
        super().__init__()

        # Default lr scheduler params
        if lr_scheduler_params is None:
            lr_scheduler_params = {
                'factor': 0.33,
                'patience': 10,
                'threshold': 1e-4,
                'cooldown': 0,
                'min_lr': 1e-6,
                'mode': 'max',
            }

        # Compute default save dir
        if save_dir is None:
            save_dir = '.'

        current_date = f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
        save_dir = f'{save_dir}/models/train/{model_name}/{current_date}'
        save_dir = os.path.normpath(save_dir)

        # Create save dir if it does not exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Initialize trackers
        self.writer = SummaryWriter(log_dir=save_dir)
        self.num_save_attempts = 0
        self.num_save_successes = 0

        self.display_reward_tracker = RewardTracker()
        self.monitor_reward_tracker = RewardTracker()
        self.best_monitored_reward = None

        self.trackers = [
            self.display_reward_tracker,
            self.monitor_reward_tracker,
        ]

        # Init lr scheduler
        self.lr_scheduler = ReduceLROnPlateau(
            optimizer=agent.optimizer,
            **lr_scheduler_params,
        )

        # Save hyperparameters
        self.env = env
        self.agent = agent
        self.update_frequency = update_frequency
        self.episode_length_start = episode_length_start
        self.episode_length_increase = episode_length_increase
        self.episode_max_length = episode_length_start
        self.save_dir = save_dir
        self.model_name = model_name
        self.save_every_n_episodes = save_every_n_episodes

        # Move to device
        self._metrics_to_device()

    def _metrics_to_device(self):
        for tracker in self.trackers:
            tracker.to(self.agent.device)

    def _train_epoch(self, num_steps: int):
        step = 0
        episode = 0

        load_bar = tqdm(total=num_steps, desc='Training', unit='step')

        while step < num_steps:
            observations = self.env.reset()
            states = observations['states']

            current_episode_reward = torch.zeros(self.env.batch_size, dtype=torch.float, device=self.agent.device)
            for j in range(int(self.episode_max_length)):
                actions = self.agent.select_action(states)
                observations = self.env.step(actions)

                states = observations['states']
                rewards = observations['rewards']
                dones = observations['dones']

                if torch.any(dones) or j == int(self.episode_max_length) - 1:
                    dones = torch.ones(self.env.batch_size, dtype=torch.bool, device=self.agent.device)

                self.agent.buffer.add('rewards', rewards)
                self.agent.buffer.add('dones', dones)

                step += 1
                load_bar.update(1)
                current_episode_reward += rewards

                if (step + 1) % self.update_frequency == 0:
                    self.agent.update()

                if torch.any(dones):
                    break

            for tracker in self.trackers:
                tracker.update(current_episode_reward)

            average_reward = self.display_reward_tracker.compute().item()
            self.writer.add_scalar('Average Reward', average_reward, episode)
            self.writer.add_scalar('Exact Reward', current_episode_reward.mean() , episode)

            if (episode + 1) % self.save_every_n_episodes == 0:
                self._save_state()

            description = f'Episode {episode} | Average Reward: {average_reward:.2f}'
            load_bar.set_description_str(description)
            episode += 1

        load_bar.close()

    def _save_state(self):
        current_average_reward = self.monitor_reward_tracker.compute()
        current_state = self.agent.policy.state_dict()

        self.writer.add_scalar(
            f'Average rewards on last {self.save_every_n_episodes} episodes',
            current_average_reward, self.num_save_attempts
        )

        # Save the last checkpoint
        torch.save(current_state, f'{self.save_dir}/last.pt')

        # Save the best checkpoint
        if self.best_monitored_reward is None or current_average_reward > self.best_monitored_reward:
            self.best_monitored_reward = current_average_reward
            torch.save(current_state, f'{self.save_dir}/best.pt')

            self.writer.add_scalar(
                'Best average reward',
                current_average_reward, self.num_save_successes
            )
            self.num_save_successes += 1

        self.lr_scheduler.step(current_average_reward)
        self.monitor_reward_tracker.reset()

        self.num_save_attempts += 1

    def train(self, num_steps: int):
        self._train_epoch(num_steps)

