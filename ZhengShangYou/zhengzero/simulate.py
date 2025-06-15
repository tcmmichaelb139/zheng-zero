from ZhengShangYou.env.zhengshangyou import ZhengShangYou
from ZhengShangYou.env.utils import create_logger

import numpy as np
from rich.console import Console
from rich.table import Table
from rich.text import Text

logger = create_logger(__name__)


def print_table(rich_table):
    """
    Generate an ascii formatted presentation of a Rich table
    Eliminates any column styling
    """
    console = Console(width=150)
    with console.capture() as capture:
        console.print(rich_table)
    return Text.from_ansi(capture.get())


def sim_model(
    players,
    num_episodes=1,
    log=True,
):
    """
    Simulate a game with the given players.
    """

    win_count = np.zeros((4, 4), dtype=int)

    for episode in range(num_episodes):
        logger.info(f"Episode {episode + 1}/{num_episodes}")
        results = sim_1_game(players, log=log)

        for i in range(4):
            win_count[results[i]][i] += 1

        logger.info("-" * 50)

    sim_results = Table(title="Simulation Results")
    sim_results.add_column("Player", justify="center")
    sim_results.add_column("1st", justify="center")
    sim_results.add_column("2nd", justify="center")
    sim_results.add_column("3rd", justify="center")
    sim_results.add_column("4th", justify="center")
    sim_results.add_column("Win %", justify="center")
    for i in range(4):
        sim_results.add_row(
            f"Player {i + 1}",
            str(win_count[i][0]),
            str(win_count[i][1]),
            str(win_count[i][2]),
            str(win_count[i][3]),
            f"{win_count[i][0] / num_episodes:.2%}",
        )

    logger.info(print_table(sim_results))


def sim_1_game(
    players,
    log=True,
):
    """
    Simulates one game
    """

    env = ZhengShangYou(
        players,
        {
            "log": log,
        },
    )

    state = env.reset()
    done = False

    logger.info("Starting a new game...")
    if log:
        logger.info("Initial hands:")
        env._print_hands()

    while not done:
        current_player = env._current_player()
        action = players[current_player].play(state)

        next_state, _, done = env.step(action)

        state = next_state

    logger.info(env._env.results)

    return env._env.results
