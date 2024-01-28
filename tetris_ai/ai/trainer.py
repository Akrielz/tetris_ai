import torch
from tqdm import tqdm

from tetris_ai.ai.env.env import Env
from tetris_ai.ai.agent_ppo import AgentPPO


class TrainerPPO:
    def __init__(
            self,
            env: Env,
            agent: AgentPPO,
            update_frequency: int = 100,
            episode_length_start: float = 100,
            episode_length_increase: float = 0,
    ):
        super().__init__()

        self.env = env
        self.agent = agent
        self.update_frequency = update_frequency
        self.episode_length_start = episode_length_start
        self.episode_length_increase = episode_length_increase
        self.episode_max_length = episode_length_start

    def _train_epoch(self, num_steps: int):
        step = 0
        episode = 0

        load_bar = tqdm(total=num_steps, desc='Training', unit='step')

        while step < num_steps:
            observations = self.env.reset()
            states = observations['states']

            current_episode_reward = torch.zeros(self.env.batch_size, dtype=torch.float, device=self.agent.device)
            for _ in range(int(self.episode_max_length)):
                actions = self.agent.select_action(states)
                observations = self.env.step(actions)

                states = observations['states']
                rewards = observations['rewards']
                dones = observations['dones']

                if torch.any(dones):
                    torch.dones = torch.ones(self.env.batch_size, dtype=torch.bool, device=self.agent.device)

                self.agent.buffer.add('rewards', rewards)
                self.agent.buffer.add('dones', dones)

                step += 1
                load_bar.update(1)
                current_episode_reward += rewards

                if (step + 1) % self.update_frequency == 0:
                    self.agent.update()

                if torch.any(dones):
                    break

            self.episode_max_length += self.episode_length_increase

            average_reward = current_episode_reward.mean().item()
            max_reward = current_episode_reward.max().item()
            min_reward = current_episode_reward.min().item()

            description = f'Episode {episode} | Average Reward: {average_reward:.2f} | Max Reward: {max_reward:.2f} | Min Reward: {min_reward:.2f}'
            load_bar.set_description_str(description)
            episode += 1

        load_bar.close()

    def train(self, num_steps: int):
        self._train_epoch(num_steps)

