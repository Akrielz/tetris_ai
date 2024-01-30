from typing import Dict, List

import numpy as np

from tetris_ai.game.actions import Action, MultiAction
from tetris_ai.game.board import Board
from tetris_ai.game.cell import Cell
from tetris_ai.game.data_types import ListVector2, Vector2
from tetris_ai.game.shape import ShapeGenerator


class TetrisEnv:
    def __init__(
            self,
            width: int = 10,
            height: int = 23,
            num_next_shapes: int = 3,
            sparse_rewards: bool = False,
            action_penalty: bool = False,
            force_down_every_n_moves: int = 10,
            force_drop_instead_of_down: bool = False,
    ):
        # Save info
        self.num_next_shapes = num_next_shapes
        self.sparse_rewards = sparse_rewards
        self.action_penalty = action_penalty

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
        self.clear_score = 0.0
        self.modifier_score = 0.0
        self.previous_score = 0.0
        self.action_score = 0.0
        self.done = False
        self.actions_since_last_placed = 0

        self.cell_dict_visual = {
            Cell.EMPTY.value: '[ ]',
            Cell.PLACED.value: '[▒]',
            Cell.PAD.value: '   ',
            Cell.CURRENT.value: '[█]',
            Cell.DISABLED.value: '[░]',
            Cell.HELD.value: '[▓]',
        }

        # Prepare force down variables
        self.force_down_every_n_steps = force_down_every_n_moves
        self.aplpy_force_down = force_down_every_n_moves > 0
        self.force_drop_instead_of_down = force_drop_instead_of_down
        self.moves_since_last_force_down = 0

    @property
    def score(self) -> float:
        return self.clear_score + self.modifier_score + self.action_score

    @property
    def action_dim(self):
        return len(Action)

    @property
    def state_dim(self):
        return self.display_matrix.shape[0], self.display_matrix.shape[1], len(Cell)

    def reset(self) -> Dict:
        self.board.reset()
        self.current_shape = self.generate_random_shape()
        self.next_shapes = [self.generate_random_shape() for _ in range(self.num_next_shapes)]
        self.hold_shape = None
        self.modifier_score = 0.0
        self.clear_score = 0.0
        self.previous_score = 0.0
        self.action_score = 0.0
        self.done = False

        self.moves_since_last_force_down = 0
        self.actions_since_last_placed = 0

        self.update_display_matrix()
        return {"state": self.display_matrix}

    def add_to_display_positions(self, positions: ListVector2, offset: Vector2, cell: Cell = Cell.PLACED):
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
            cell = Cell.HELD if self.can_swap else Cell.DISABLED
            self.add_to_display_positions(self.hold_shape.shape_id.blocks_position, self.held_shape_offset, cell)

        # Display the current shape
        self.add_to_display_positions(self.current_shape.blocks_position, self.board_offset, Cell.CURRENT)

        # Display the next shapes
        for i, shape in enumerate(self.next_shapes):
            self.add_to_display_positions(
                shape.shape_id.blocks_position,
                self.next_shape_offset + np.array([i * (2 + 1), 0]),
                Cell.HELD
            )

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

        self.clear_score += num_cleared_lines ** 2

    def check_defeat(self):
        positions = self.current_shape.blocks_position
        self.done = True if not self.board.are_free(positions) else False
        self.clear_score -= 10.0 if self.done else 0.0

    def add_score_place_modifiers(self):
        self.modifier_score = 0.0

        # Add a small bonus for each block placed correctly
        for _ in self.current_shape.blocks_position:
            self.modifier_score += 0.01

        # Add a penalty for each block placed too high
        for block_position in self.current_shape.blocks_position:
            y, x = block_position
            tallness = self.board.height - y
            if tallness >= self.board.height // 2:
                tall_penalty = tallness - self.board.height // 2
                self.modifier_score -= tall_penalty * 0.003

        # Add a penalty for each block placed incorrectly
        xs = set(self.current_shape.blocks_position[:, 1])
        ys_max_per_xs = {}
        ys_min_per_xs = {}
        for x in xs:
            mask = self.current_shape.blocks_position[:, 1] == x
            ys_max_per_xs[x] = np.min(self.current_shape.blocks_position[mask][:, 0])
            ys_min_per_xs[x] = np.max(self.current_shape.blocks_position[mask][:, 0])

        for x in xs:
            y = ys_max_per_xs[x]
            for i in range(y+1, self.board.height):
                if self.board[i, x] != Cell.EMPTY.value:
                    break
                self.modifier_score -= 0.1

        # Add penalty for bumpy terrain
        # TODO: Implement this

    def place_shape(self):
        self.current_shape.place()
        self.add_score_place_modifiers()
        self.check_clear_lines()
        self.current_shape = self.next_shapes.pop(0)
        self.next_shapes.append(self.generate_random_shape())
        self.check_defeat()
        self.can_swap = True
        self.actions_since_last_placed = 0
        self.moves_since_last_force_down = 0

    def process_action(self, action: Action):
        self.moves_since_last_force_down += 1

        if action == Action.LEFT:
            self.current_shape.move(np.array([0, -1]))
            return

        if action == Action.RIGHT:
            self.current_shape.move(np.array([0, 1]))
            return

        if action == Action.DOWN:
            if self.current_shape.can_move_down():
                self.current_shape.move(np.array([1, 0]))
            else:
                self.place_shape()
            self.moves_since_last_force_down = 0
            return

        if action == Action.ROTATE:
            self.current_shape.rotate(is_clockwise=True)
            return

        if action == Action.DROP:
            while self.current_shape.can_move_down():
                self.current_shape.move(np.array([1, 0]))
            self.place_shape()
            self.moves_since_last_force_down = 0
            return

        if action == Action.NOOP:
            return

        if action == Action.SWAP:
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
            self.actions_since_last_placed = 0
            self.moves_since_last_force_down = 0
            return

    def prepare_outputs(self) -> Dict:
        state = self.display_matrix
        reward = self.score
        done = self.done
        info = {}

        output = {
            'state': state,
            'reward': reward,
            'done': done,
            'info': info,
        }
        return output

    def apply_sparse_rewards(self):
        if not self.sparse_rewards:
            return

        self.previous_score = self.score
        self.clear_score = 0.0
        self.modifier_score = 0.0
        self.action_score = 0.0

    def apply_action_penalty(self, action: Action):
        if not self.action_penalty:
            return

        self.action_score = 0.0

        if action == Action.NOOP:
            self.action_score -= 0.1

        elif action == Action.SWAP:
            if self.actions_since_last_placed > 0:
                self.action_score -= 0.1

        elif action == Action.DOWN:
            pass

        elif action == Action.LEFT:
            pass

        elif self.actions_since_last_placed > 2 * self.board.width:
            self.action_score -= 0.01

        self.actions_since_last_placed += 1

    def apply_force_move_down(self):
        if not self.aplpy_force_down:
            return

        if self.moves_since_last_force_down >= self.force_down_every_n_steps:
            if self.force_drop_instead_of_down:
                self.process_action(Action.DROP)
            else:
                self.process_action(Action.DOWN)

    def step(self, action: Action | int) -> Dict:
        if isinstance(action, int):
            action = Action(action)

        self.process_action(action)
        self.apply_force_move_down()
        self.apply_action_penalty(action)
        self.update_display_matrix()

        outputs = self.prepare_outputs()
        self.apply_sparse_rewards()

        return outputs

    def render_console(self):
        # Clear screen the previous frame
        print('\n' * 80)

        # Update the display matrix
        self.update_display_matrix()
        display_string = "\n".join(["".join([self.cell_dict_visual[col] for col in row]) for row in self.display_matrix])

        # Print the display matrix
        print(display_string)

        print("Score: ", self.previous_score)


class MultiActionTetrisEnv(TetrisEnv):
    def __init__(
            self,
            width: int = 10,
            height: int = 23,
            num_next_shapes: int = 3,
            sparse_rewards: bool = False,
            action_penalty: bool = False,
            force_down_every_n_moves: int = 0,
            force_drop_instead_of_down: bool = False,
    ):
        super().__init__(
            width,
            height,
            num_next_shapes,
            sparse_rewards,
            action_penalty,
            force_down_every_n_moves,
            force_drop_instead_of_down,
        )

        self.multi_action = MultiAction(width)

    @property
    def action_dim(self):
        return len(self.multi_action.action_space())

    def step(self, action: int) -> Dict:
        actions = self.multi_action(action)
        return self.multi_step(actions)

    def multi_step(self, actions: List[Action | int]) -> Dict:
        for action in actions:
            observations = super().step(action)

        return observations
