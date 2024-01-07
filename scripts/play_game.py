import random

from tetris_ai.game.actions import Action
from tetris_ai.game.tetris import TetrisEnv


def get_player_action():
    input_action = input("Action: ")
    input_map = {
        "a": Action.LEFT,
        "d": Action.RIGHT,
        "s": Action.DOWN,
        "w": Action.ROTATE,
        " ": Action.DROP,
        "n": Action.NOOP,
        'q': Action.SWAP,
    }

    return input_map.get(input_action, Action.NOOP)


def get_random_action():
    return random.choice(Action.action_space())


def main():
    env = TetrisEnv(height=10)
    env.reset()

    while True:
        env.render_console()
        action = get_player_action()
        state, reward, done, info = env.step(action)

        if done:
            env.reset()


if __name__ == "__main__":
    main()