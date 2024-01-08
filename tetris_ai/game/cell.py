from enum import Enum


class Cell(Enum):
    """
    Cell types that can be found in the board.

    EMPTY: Empty cell.
    PLACED: Cell that has been placed.
    PAD: Padding cell.
    CURRENT: Cell that is part of the current shape.
    DISABLED: Cell that is part of the current shape but cannot be placed.
    HELD: Cell that is part of the held shape.
    """

    EMPTY = 0
    PLACED = 1
    PAD = 2
    CURRENT = 3
    DISABLED = 4
    HELD = 5

    def __str__(self):
        return self.name.lower()

    def __repr__(self):
        return self.__str__()

    @staticmethod
    def action_space():
        return [Cell.EMPTY, Cell.PLACED, Cell.PAD, Cell.CURRENT, Cell.DISABLED]