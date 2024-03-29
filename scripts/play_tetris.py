import random

from tetris_ai.game.actions import Action, MultiAction
from tetris_ai.game.tetris import TetrisEnv, MultiActionTetrisEnv


def get_player_action():
    input_action = input("Action: ")[0]
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


def get_player_multi_action(multi_action: MultiAction):
    input_action = input(f"Action [0-{len(multi_action.action_space())}]: ")
    input_action_int = int(input_action)
    return input_action_int


def get_random_action():
    return random.choice(Action.action_space())


def play_tetris():
    env = TetrisEnv(height=23, sparse_rewards=True, force_down_every_n_moves=4, force_drop_instead_of_down=False)
    env.reset()

    score = 0.0
    while True:
        env.render_console()
        print(f"Cumulative Score: {score}")
        action = get_player_action()
        output = env.step(action)
        done = output['done']
        score += output['reward']

        if done:
            env.reset()
            score = 0


def play_tetris_multi_action():
    env = MultiActionTetrisEnv(height=10, sparse_rewards=True)
    env.reset()

    while True:
        env.render_console()
        actions = get_player_multi_action(env.multi_action)
        print(actions)
        output = env.step(actions)
        done = output['done']

        if done:
            env.reset()


if __name__ == "__main__":
    play_tetris()
