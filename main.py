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
    }

    return input_map.get(input_action, Action.NOOP)


def main():
    env = TetrisEnv(clock_speed=1000, height=10)

    while True:
        env.render_console()
        action = get_player_action()
        state, reward, done, info = env.step(action)


if __name__ == "__main__":
    main()