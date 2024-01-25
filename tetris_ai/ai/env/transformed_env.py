from typing import Optional

from tetris_ai.ai.env.env import Env
from tetris_ai.ai.env.torch_env import TorchEnv
from tetris_ai.ai.env.transform import Transform, Identity


class TransformedEnv(Env):
    def __init__(
            self,
            env: TorchEnv,
            transform: Optional[Transform] = None,
    ):
        super().__init__()

        self.env = env
        self.transform = transform

        if self.transform is None:
            self.transform = Identity()

    def reset(self):
        state = self.env.reset()
        state = self.transform(state)
        return state

    def step(self, action):
        output = self.env.step(action)
        output = self.transform(output)
        return output

    def rand_step(self):
        output = self.env.rand_step()
        output = self.transform(output)
        return output

    @property
    def action_dim(self):
        return self.env.action_dim

    @property
    def state_dim(self):
        return self.env.state_dim
