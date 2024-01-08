import numpy as np

from tetris_ai.game.actions import Action
from tetris_ai.game.board import Board
from tetris_ai.game.cell import Cell
from tetris_ai.game.data_types import ListVector2, Vector2
from tetris_ai.game.shape import ShapeGenerator


class TetrisEnv:
    def __init__(
            self,
            width: int = 10,
            height: int = 22,
            num_next_shapes: int = 3,
    ):
        # Save clock speed
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
        self.start_position = np.array([1, width // 2])

        self.current_shape = self.generate_random_shape()
        self.next_shapes = [self.generate_random_shape() for _ in range(num_next_shapes)]
        self.hold_shape = None
        self.can_swap = True

        # Prepare the score
        self.score = 0
        self.done = False

    def reset(self):
        self.board.reset()
        self.current_shape = self.generate_random_shape()
        self.next_shapes = [self.generate_random_shape() for _ in range(self.num_next_shapes)]
        self.hold_shape = None
        self.score = 0
        self.done = False

        self.update_display_matrix()
        return self.display_matrix  # return the state

    def add_to_display_positions(self, positions: ListVector2, offset: Vector2, cell: Cell = Cell.FULL):
        self.display_matrix[positions[:, 0] + offset[0], positions[:, 1] + offset[1]] = cell.value

    def update_display_matrix(self):
        self.display_matrix.fill(Cell.PAD.value)

        # Display the board
        board_state = self.board.state
        start_board_x = self.board_offset[1]
        stop_board_x = board_state.shape[1] + self.board_offset[1]
        self.display_matrix[:, start_board_x: stop_board_x] = board_state

        # Display the held shape
        if self.hold_shape is not None:
            cell = Cell.FULL if self.can_swap else Cell.DISABLED
            self.add_to_display_positions(self.hold_shape.shape_id.blocks_position, self.held_shape_offset, cell)

        # Display the current shape
        self.add_to_display_positions(self.current_shape.blocks_position, self.board_offset, Cell.CURRENT)

        # Display the next shapes
        for i, shape in enumerate(self.next_shapes):
            self.add_to_display_positions(shape.shape_id.blocks_position, self.next_shape_offset + np.array([i * (2 + 1), 0]))

    def generate_random_shape(self):
        return self.shape_generator.generate_random_shape(self.start_position)

    def check_clear_lines(self):
        ys = self.current_shape.blocks_position[:, 0]

        num_cleared_lines = 0
        for y in ys:
            if np.all(self.board[y, :] == 1):
                self.board[y, :] = 0
                self.board[1:y+1, :] = self.board[:y, :]
                self.board[0, :] = 0
                num_cleared_lines += 1

        self.score += num_cleared_lines ** 2

    def check_defeat(self):
        positions = self.current_shape.blocks_position
        self.done = True if not self.board.are_free(positions) else False

    def place_shape(self):
        self.current_shape.place()
        self.check_clear_lines()
        self.current_shape = self.next_shapes.pop(0)
        self.next_shapes.append(self.generate_random_shape())
        self.check_defeat()
        self.can_swap = True

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
            if not self.can_swap:
                return

            if self.hold_shape is None:
                self.current_shape.reset()
                self.hold_shape = self.current_shape
                self.current_shape = self.generate_random_shape()
            else:
                self.current_shape.reset()
                self.hold_shape.reset()
                self.current_shape, self.hold_shape = self.hold_shape, self.current_shape

            self.can_swap = False

    def prepare_outputs(self):
        state = self.display_matrix
        reward = self.score
        done = self.done
        info = {}

        return state, reward, done, info

    def step(self, action: Action):
        self.process_action(action)
        self.update_display_matrix()

        return self.prepare_outputs()

    def render_console(self):
        # Clear screen the previous frame
        print('\n' * 80)

        # Update the display matrix
        self.update_display_matrix()
        display_string = ""
        for row in self.display_matrix:
            for col in row:
                if col == Cell.EMPTY.value:
                    display_string += "[ ]"
                elif col == Cell.FULL.value:
                    display_string += "[█]"
                elif col == Cell.PAD.value:
                    display_string += "   "
                elif col == Cell.CURRENT.value:
                    display_string += "[▓]"
                elif col == Cell.DISABLED.value:
                    display_string += "[░]"
            display_string += "\n"

        # Print the display matrix
        print(display_string)

        print("Score: ", self.score)

        # sleep the ammount of time specified by the clock speed
        # sleep(self.clock_speed / 1000)