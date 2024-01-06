import numpy as np

from tetris_ai.game.board import Board
from tetris_ai.game.data_types import ListVector2, Vector2, rotate_points


class ShapeID:
    def __init__(
            self,
            blocks_position: ListVector2,
            rotation_point: Vector2,
    ):
        self.blocks_position = blocks_position
        self.rotation_point = rotation_point


class Shape:
    def __init__(
            self,
            board: Board,
            position: Vector2,
            shape_id: ShapeID,
    ):
        self.board = board
        self.current_position = position.copy()
        self.start_position = position.copy()
        self.shape_id = shape_id
        self.dead = False
        self.blocks_position = self.shape_id.blocks_position + self.current_position

    def can_move(self, direction: Vector2) -> bool:
        new_positions = self.blocks_position + direction
        return self.board.are_free(new_positions)

    def can_move_down(self) -> bool:
        return self.can_move(np.array([1, 0]))

    def move_shape(self, direction: Vector2):
        if not self.can_move(direction):
            return

        self.current_position += direction
        self.blocks_position = self.blocks_position + direction

    def simulate_rotation(self, is_clockwise: bool) -> ListVector2:
        block_positions = self.blocks_position.copy()
        starting_positions = block_positions - self.current_position

        rotation_angle = 90 if is_clockwise else -90

        rotated_positions = rotate_points(
            points=starting_positions,
            degrees=rotation_angle,
            origin=self.shape_id.rotation_point,
        )

        rotated_positions += self.current_position

        rounded_rotated_positions = np.round(rotated_positions)
        return rounded_rotated_positions

    def can_rotate(self, is_clockwise: bool) -> bool:
        rotated_positions = self.simulate_rotation(is_clockwise)
        return self.board.are_free(rotated_positions)

    def place_shape(self):
        self.board[self.blocks_position[:, 0], self.blocks_position[:, 1]] = 1
        self.dead = True

    def rotate(self, is_clockwise: bool):
        if not self.can_rotate(is_clockwise):
            return

        self.blocks_position = self.simulate_rotation(is_clockwise)

    def __repr__(self):
        visual_matrix = np.zeros((4, 4))
        relative_position = self.blocks_position - self.current_position
        visual_matrix[relative_position[:, 0], relative_position[:, 1]] = 1
        string_representation = ""
        for row in visual_matrix:
            for col in row:
                if col == 0:
                    string_representation += "[ ]"
                else:
                    string_representation += "[█]"
            string_representation += "\n"

        return string_representation


class ShapeGenerator:

    def __init__(self, board: Board):
        self.board = board
        self.shapes = self.generate_internal_shapes()
        self.shape_names = list(self.shapes.keys())

    @staticmethod
    def generate_internal_shapes():

        # [XX]
        # [XX]
        shape_square = ShapeID(
            blocks_position=np.array([
                [0, 0],
                [1, 0],
                [0, 1],
                [1, 1],
            ]),
            rotation_point=np.array([0.5, 0.5]),
        )

        # [  X]
        # [XXX]
        shape_l = ShapeID(
            blocks_position=np.array([
                [0, 2],
                [1, 0],
                [1, 1],
                [1, 2],
            ]),
            rotation_point=np.array([1.0, 0.0]),
        )

        # [x  ]
        # [XXX]
        shape_reverse_l = ShapeID(
            blocks_position=np.array([
                [0, 0],
                [1, 0],
                [1, 1],
                [1, 2],
            ]),
            rotation_point=np.array([0.0, 1.0]),
        )

        # [XXXX]
        shape_line = ShapeID(
            blocks_position=np.array([
                [0, 0],
                [0, 1],
                [0, 2],
                [0, 3],
            ]),
            rotation_point=np.array([0.5, 1.5]),
        )

        # [ X ]
        # [XXX]
        shape_t = ShapeID(
            blocks_position=np.array([
                [0, 1],
                [1, 0],
                [1, 1],
                [1, 2],
            ]),
            rotation_point=np.array([1.0, 1.0]),
        )

        # [XX ]
        # [ XX]
        shape_z = ShapeID(
            blocks_position=np.array([
                [0, 0],
                [0, 1],
                [1, 1],
                [1, 2],
            ]),
            rotation_point=np.array([1.0, 1.0]),
        )

        # [ XX]
        # [XX ]
        shape_s = ShapeID(
            blocks_position=np.array([
                [1, 0],
                [1, 1],
                [0, 1],
                [0, 2],
            ]),
            rotation_point=np.array([0.0, 1.0]),
        )

        shapes = {
            'square': shape_square,
            'l': shape_l,
            'reverse_l': shape_reverse_l,
            'line': shape_line,
            't': shape_t,
            'z': shape_z,
            's': shape_s,
        }

        return shapes

    def generate_random_shape(self, position: Vector2) -> Shape:
        shape_name = np.random.choice(self.shape_names)
        shape_id = self.shapes[shape_name]
        return Shape(
            board=self.board,
            position=position,
            shape_id=shape_id,
        )
