import numpy as np

from tetris_ai.game.data_types import Vector2


class Board:
    def __init__(
            self,
            width: int,
            height: int
    ):
        self.width = width
        self.height = height
        self._placed_squares = np.zeros((self.height, self.width))

    def __getitem__(self, item):
        return self._placed_squares[item]

    def __setitem__(self, key, value):
        self._placed_squares[key] = value

    def is_free(self, position: Vector2):
        y, x = position

        if y < 0 or y >= self.height:
            return False

        if x < 0 or x >= self.width:
            return False

        return self._placed_squares[y, x] == 0

    def are_free(self, positions: np.ndarray):
        if np.any(positions[:, 0] < 0) or np.any(positions[:, 0] >= self.height):
            return False

        if np.any(positions[:, 1] < 0) or np.any(positions[:, 1] >= self.width):
            return False

        return np.all(self._placed_squares[positions[:, 0], positions[:, 1]] == 0)

    @property
    def state(self):
        return self._placed_squares