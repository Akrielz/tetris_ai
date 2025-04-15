import pygame

from tetris_ai.game.cell import Cell
from tetris_ai.game.tetris import TetrisEnv


class VisualTetrisEnv:
    def __init__(self, env: TetrisEnv, block_dim: int = 100, offset_space: int = 2):
        pygame.init()
        self.env = env

        self.display_width = self.env.display_matrix.shape[1] * (block_dim + offset_space)
        self.display_height = self.env.display_matrix.shape[0] * (block_dim + offset_space)

        self.block_dim = block_dim
        self.offset_space = offset_space

        self.screen = pygame.display.set_mode((self.display_width, self.display_height))

        self.cell_dict_colors = {
            Cell.EMPTY.value: (0, 0, 0),
            Cell.PLACED.value: (127, 127, 127),
            Cell.PAD.value: (10, 10, 10),
            Cell.CURRENT.value: (255, 0, 0),
            Cell.DISABLED.value: (127, 127, 127),
            Cell.HELD.value: (0, 0, 255),
        }

    def render(self):
        self.screen.fill((0, 0, 0))
        for y, row in enumerate(self.env.display_matrix):
            for x, cell in enumerate(row):
                x_start = x * (self.block_dim + self.offset_space)
                y_start = y * (self.block_dim + self.offset_space)
                rect = (x_start, y_start, self.block_dim, self.block_dim)

                pygame.draw.rect(
                    self.screen,
                    self.cell_dict_colors[cell],
                    rect
                )

        pygame.display.flip()
