import torch
from tqdm import tqdm

from tetris_ai.ai.ppo_agent import PPOAgent
from tetris_ai.game.tetris import TetrisEnv


class TrainerPPO:
    def __init__(
            self,
            env: TetrisEnv,
            agent: PPOAgent,
            max_episode_length: int = 1000,
            update_frequency: int = 4,
    ):
        super().__init__()

        self.env = env
        self.agent = agent
        self.max_episode_length = max_episode_length
        self.update_frequency = update_frequency

    def _train_epoch(self, num_steps: int):
        step = 0
        episode = 0

        load_bar = tqdm(total=num_steps, desc='Training', unit='step')

        while step < num_steps:
            state = self.env.reset()
            current_episode_reward = 0

            for _ in range(self.max_episode_length):
                action = self.agent.select_action(state)
                next_state, reward, done, _ = self.env.step(action)

                self.agent.buffer.add('rewards', reward)
                self.agent.buffer.add('dones', done)

                step += 1
                load_bar.update(1)
                current_episode_reward += reward

                if step % self.update_frequency == 0:
                    self.agent.update()

                if done:
                    break

            print(f'Episode {episode} reward: {current_episode_reward}')
            episode += 1

        load_bar.close()

    def train(self, num_steps: int):
        # TODO: Add batch_size
        self._train_epoch(num_steps)

