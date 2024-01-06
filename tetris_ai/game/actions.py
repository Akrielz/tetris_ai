from enum import Enum


class Action(Enum):
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

    LEFT = 0
    RIGHT = 1
    DOWN = 2
    ROTATE = 3
    DROP = 4
    NOOP = 5
    SWAP = 6

    def __str__(self):
        return self.name.lower()

    def __repr__(self):
        return self.__str__()

    @staticmethod
    def action_space():
        return [Action.LEFT, Action.RIGHT, Action.DOWN, Action.ROTATE, Action.DROP, Action.NOOP, Action.SWAP]