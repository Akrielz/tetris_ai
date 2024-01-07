from time import sleep

import numpy as np

from tetris_ai.game.actions import Action
from tetris_ai.game.board import Board
from tetris_ai.game.data_types import ListVector2, Vector2
from tetris_ai.game.shape import ShapeGenerator


class TetrisEnv:
    def __init__(
            self,
            width: int = 10,
            height: int = 22,
            clock_speed: int = 0,  # in milliseconds
            num_next_shapes: int = 3,
    ):
        # Save clock speed
        self.clock_speed = clock_speed
        self.num_next_shapes = num_next_shapes

        # Create the displayable parts
        self.held_shape_display = np.zeros((2, 4))
        self.board = Board(width, height)
        self.next_shape_display = np.zeros(((2 + 1) * num_next_shapes, 4))

        # Compute offsets for display
        self.held_shape_offset = np.array([1, 0])
        self.board_offset = np.array([0, 5])  # len of shape + 1
        self.next_shape_offset = np.array([1, self.board_offset[1] + width + 1])

        self.display_width = self.held_shape_display.shape[1] + 1 + width + 1 + self.next_shape_display.shape[1]

        self.display_matrix = np.zeros((height, self.display_width))

        # Prepare the pieces
        self.shape_generator = ShapeGenerator(self.board)
        self.start_position = np.array([0, width // 2])

        self.current_shape = self.generate_random_shape()
        self.next_shapes = [self.generate_random_shape() for _ in range(num_next_shapes)]
        self.hold_shape = None

    def add_to_display_positions(self, positions: ListVector2, offset: Vector2):
        self.display_matrix[positions[:, 0] + offset[0], positions[:, 1] + offset[1]] = 1

    def update_display_matrix(self):
        self.display_matrix.fill(2)

        # Display the board
        board_state = self.board.state
        start_board_x = self.board_offset[1]
        stop_board_x = board_state.shape[1] + self.board_offset[1]
        self.display_matrix[:, start_board_x: stop_board_x] = board_state

        # Display the held shape
        if self.hold_shape is not None:
            self.add_to_display_positions(self.hold_shape.shape_id.blocks_position, self.held_shape_offset)

        # Display the current shape
        self.add_to_display_positions(self.current_shape.blocks_position, self.board_offset)

        # Display the next shapes
        for i, shape in enumerate(self.next_shapes):
            self.add_to_display_positions(shape.shape_id.blocks_position, self.next_shape_offset + np.array([i * (2 + 1), 0]))

    def generate_random_shape(self):
        return self.shape_generator.generate_random_shape(self.start_position)

    def clear_lines(self):
        ys = self.current_shape.blocks_position[:, 0]
        for y in ys:
            if np.all(self.board[y, :] == 1):
                self.board[y, :] = 0
                self.board[1:y+1, :] = self.board[:y, :]
                self.board[0, :] = 0

    def place_shape(self):
        self.current_shape.place()
        self.clear_lines()

        self.current_shape = self.next_shapes.pop(0)
        self.next_shapes.append(self.generate_random_shape())

    def process_action(self, action: Action):
        if action == Action.LEFT:
            self.current_shape.move(np.array([0, -1]))

        elif action == Action.RIGHT:
            self.current_shape.move(np.array([0, 1]))

        elif action == Action.DOWN:
            if self.current_shape.can_move_down():
                self.current_shape.move(np.array([1, 0]))
            else:
                self.place_shape()

        elif action == Action.ROTATE:
            self.current_shape.rotate(is_clockwise=True)

        elif action == Action.DROP:
            while self.current_shape.can_move_down():
                self.current_shape.move(np.array([1, 0]))
            self.place_shape()

        elif action == Action.NOOP:
            pass

        elif action == Action.SWAP:
            if self.hold_shape is None:
                self.current_shape.reset()
                self.hold_shape = self.current_shape
                self.current_shape = self.generate_random_shape()
            else:
                self.current_shape.reset()
                self.hold_shape.reset()
                self.current_shape, self.hold_shape = self.hold_shape, self.current_shape

    def step(self, action: Action):
        self.process_action(action)
        self.update_display_matrix()

        state = self.display_matrix
        reward = 0
        done = False
        info = {}

        return state, reward, done, info

    def render_console(self):
        # Clear screen the previous frame
        print('\n' * 80)

        # Update the display matrix
        self.update_display_matrix()
        display_string = ""
        for row in self.display_matrix:
            for col in row:
                if col == 0:
                    display_string += "[ ]"
                elif col == 1:
                    display_string += "[█]"
                else:
                    display_string += "   "
            display_string += "\n"

        # Print the display matrix
        print(display_string)

        # sleep the ammount of time specified by the clock speed
        # sleep(self.clock_speed / 1000)