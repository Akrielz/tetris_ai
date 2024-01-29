from enum import Enum


class LimitedAction(Enum):
    """
    Limited set of actions that can be taken by the agent.

    LEFT: Move the piece left.
    RIGHT: Move the piece right.
    ROTATE: Rotate the piece.
    DROP: Drop the piece.
    SWAP: Swap the current piece with the held piece (if possible).
    """

    LEFT = 0
    RIGHT = 1
    ROTATE = 2
    DROP = 3
    SWAP = 4

    def __str__(self):
        return self.name.lower()

    def __repr__(self):
        return self.__str__()

    @staticmethod
    def action_space():
        return [LimitedAction.LEFT, LimitedAction.RIGHT, LimitedAction.ROTATE, LimitedAction.DROP, Action.SWAP]


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
    ROTATE = 2
    DROP = 3
    SWAP = 4
    DOWN = 5
    NOOP = 6

    def __str__(self):
        return self.name.lower()

    def __repr__(self):
        return self.__str__()

    @staticmethod
    def action_space():
        return [Action.LEFT, Action.RIGHT, Action.ROTATE, Action.DROP, Action.SWAP, Action.DOWN, Action.NOOP]


class MultiAction:
    def __init__(self, width):
        self.width = width
        self.multi_actions_list = self._prepare_multi_actions(width)

    @staticmethod
    def _prepare_multi_actions(width):
        multi_actions_list = []
        for num_rotations in range(4):
            for num_columns in range(width):
                actions = []

                # Add that many ROTATE
                for _ in range(num_rotations):
                    actions.append(Action.ROTATE)

                # Add half the width go to LEFT, to ensure the piece is at the left edge
                for _ in range(width // 2 + 1):
                    actions.append(Action.LEFT)

                # Add that many RIGHT to go to the correct column
                for _ in range(num_columns):
                    actions.append(Action.RIGHT)

                # Add DROP to drop the piece
                actions.append(Action.DROP)

                multi_actions_list.append(actions)

        return multi_actions_list

    def __call__(self, mega_action_index):
        return self.multi_actions_list[mega_action_index]

    def action_space(self):
        return [i for i in range(len(self.multi_actions_list))]

