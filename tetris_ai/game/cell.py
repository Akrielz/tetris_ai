from enum import Enum


class Cell(Enum):
    """
    Actions that can be taken by the agent.

    LEFT: Move the piece left.
    RIGHT: Move the piece right.
    DOWN: Move the piece down.
    ROTATE: Rotate the piece.
    DROP: Drop the piece.
    NOOP: Do nothing.
    SWAP: Swap the current piece with the held piece (if possible).
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