from vars import GLOBAL_LOG_FOLDER

from ZhengShangYou.env.zhengshangyou import ZhengShangYou
from ZhengShangYou.env.utils import (
    create_logger,
)

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque
from copy import deepcopy

import trueskill

logger = create_logger(__name__)


def save_rewards(all_rewards):
    plt.figure()

    def moving_stat(data, window_size):
        if len(data) < window_size:
            return np.array([np.mean(data)]), np.array(
                [np.std(data) / np.sqrt(len(data)) if len(data) > 0 else 0]
            )

        avg = np.convolve(data, np.ones(window_size), "valid") / window_size

        std = np.array(
            [
                np.std(data[i : i + window_size]) / np.sqrt(window_size)
                for i in range(len(data) - window_size + 1)
            ]
        )

        return avg, std

    if len(all_rewards) == 0 or len(all_rewards[0]) < 100:
        return

    for i, rewards in enumerate(all_rewards):
        window_size = 100
        avg_rewards, std_rewards = moving_stat(rewards, window_size)

        x_values = np.arange(window_size, len(rewards) + 1, 1)

        sns.lineplot(x=x_values, y=avg_rewards, label=f"Player {i}")

        lower_bound = avg_rewards - std_rewards
        upper_bound = avg_rewards + std_rewards

        plt.fill_between(
            x_values,
            lower_bound,
            upper_bound,
            alpha=0.2,
        )

    plt.xlabel("Episodes")
    plt.ylabel("Average Reward")
    plt.title("Average Reward vs Episodes")
    plt.legend()
    plt.show()
    plt.savefig(GLOBAL_LOG_FOLDER + "/average_reward.png")

    import json

    with open(GLOBAL_LOG_FOLDER + "/rewards.json", "w") as f:
        json.dump(
            {f"Player {i}": rewards for i, rewards in enumerate(all_rewards)},
            f,
        )


def save_skills(skills):
    if len(skills) == 0:
        return

    plt.figure()

    sns.lineplot(x=np.arange(len(skills)), y=skills, label="Player Skill")
    plt.xlabel("Episodes")
    plt.ylabel("Skill (TrueSkill Mu)")
    plt.title("Player Skill vs Episodes")
    plt.legend()
    plt.show()
    plt.savefig(GLOBAL_LOG_FOLDER + "/skill.png")

    import json

    with open(GLOBAL_LOG_FOLDER + "/skills.json", "w") as f:
        json.dump({"Player Skill": skills}, f)


def train_model(
    players,
    zhengzero_player_ids,
    batch_size,
    params,
):
    """
    Train the model with the given parameters.
    useful when training against random and algorithmic bots
    :param players: The players to be trained
    :param batch_size: The batch size
    :param num_episodes: The number of episodes to train
    """

    episode = 1
    all_rewards = [[] for _ in range(len(zhengzero_player_ids))]
    win_count = np.zeros((len(players), len(players)), dtype=int)
    win_count_100 = np.zeros((len(players), len(players)), dtype=int)

    try:
        env = ZhengShangYou(players, params)

        while True:
            logger.info(f"##### Episode {episode + 1} #####")

            state = env.reset()
            total_reward = [-100] * len(zhengzero_player_ids)
            done = False

            while not done:
                current_player = env._current_player()
                action = players[current_player].play(state)

                next_state, reward, done = env.step(action)

                players[current_player].remember(state, action, reward, False)

                state = next_state

                if current_player in zhengzero_player_ids:
                    total_reward[zhengzero_player_ids.index(current_player)] = max(
                        total_reward[zhengzero_player_ids.index(current_player)],
                        reward,
                    )

            final_rewards = env.get_final_rewards()

            for i, ids in enumerate(zhengzero_player_ids):
                total_reward[i] += final_rewards[ids]
                all_rewards[i].append(total_reward[i])

            for i in zhengzero_player_ids:
                players[i].remember(None, None, final_rewards[i], True)
                players[i].replay(batch_size, episode + 1)

            for i, j in enumerate(env._env.results):
                win_count_100[j][i] += 1

            if (episode + 1) % 100 == 0:
                for i in range(len(players)):
                    for j in range(len(players)):
                        win_count[i][j] += win_count_100[i][j]

                for i in range(len(players)):
                    wins = " ".join(
                        [
                            f"{win_count[i][j]} ({win_count_100[i][j]})"
                            for j in range(len(players))
                        ]
                    )

                    logger.info(
                        f"Player {i} wins: [{wins}]",
                    )

                win_count_100 = np.zeros((len(players), len(players)), dtype=int)

            episode += 1

    except KeyboardInterrupt:
        save_rewards(all_rewards)
    except Exception as e:
        logger.exception(e)


def self_play(player, pool_size=10, batch_size=100, update_length=1000):
    """
    Self-play training for the player
    :param player: The player to be trained
    """

    ts = trueskill.TrueSkill(draw_probability=0.0)

    opponent_pool = []

    rewards = []
    skills = []

    try:
        for _ in range(3):
            opp = player.clone()
            opp.params["train"] = False
            opponent_pool.append(opp)

        episode = 1

        win_count = np.zeros((4), dtype=int)
        win_count_100 = np.zeros((4), dtype=int)

        while True:
            logger.info(f"##### Self-play Episode {episode} #####")

            players = [player]
            chosen = []
            for i in range(3):
                idx = np.random.choice(len(opponent_pool))
                while idx in chosen:
                    idx = np.random.choice(len(opponent_pool))
                chosen.append(idx)
                opp = opponent_pool[idx]
                opp.player_id = i + 1
                players.append(opp)

            env = ZhengShangYou(players, {"train": True, "log": False})

            state = env.reset()

            total_reward = -100

            done = False

            while not done:
                current_player = env._current_player()
                action = players[current_player].play(state)

                next_state, reward, done = env.step(action)

                players[current_player].remember(state, action, reward, False)

                state = next_state

                if current_player == 0:
                    total_reward = max(total_reward, reward)

            final_rewards = env.get_final_rewards()
            total_reward += final_rewards[0]
            rewards.append(total_reward)

            players[0].remember(None, None, final_rewards[0], True)
            players[0].replay(batch_size, episode)

            win_count_100[env._env.results.index(0)] += 1

            # update player ratings using TrueSkill
            results = env._env.results
            ranks = [0] * 4
            for i, p in enumerate(results):
                ranks[p] = i

            rating_group = [(players[i].rating,) for i in range(len(players))]
            new_ratings = ts.rate(rating_group, ranks)

            for i, p in enumerate(players):
                p.rating = new_ratings[i][0]

            if episode % 100 == 0:
                for i in range(4):
                    win_count[i] += win_count_100[i]

                wins = " ".join(
                    [
                        f"{win_count[j]} ({win_count_100[j]})"
                        for j in range(len(players))
                    ]
                )

                logger.info(
                    f"Player {i} wins: [{wins}]",
                )

                win_count_100 = np.zeros((4), dtype=int)

                skills.append(player.rating.mu)
                logger.info(f"Player skill (TrueSkill Mu): {player.rating.mu}")

            if episode % update_length == 0:
                logger.info("Updating opponent pool...")

                new_opponent = player.clone(episode)
                new_opponent.params["train"] = False

                if len(opponent_pool) < pool_size:
                    opponent_pool.append(new_opponent)
                else:
                    opponent_pool.sort(key=lambda x: x.rating.mu, reverse=True)
                    if new_opponent.rating.mu > opponent_pool[-1].rating.mu:
                        opponent_pool[-1] = new_opponent

            episode += 1

    except KeyboardInterrupt:
        save_rewards([rewards])
        save_skills(skills)

        logger.info("Top opponent pool:")

        opponent_pool.sort(key=lambda x: x.rating.mu, reverse=True)

        top_opponent_dir = GLOBAL_LOG_FOLDER + "/top_opponent_pool/"

        os.makedirs(top_opponent_dir, exist_ok=True)

        for i, opp in enumerate(opponent_pool):
            logger.info(f"Opponent {i}: {opp.player_name}, Skill: {opp.rating.mu}")

            opp.model_dir = top_opponent_dir
            opp.save_model(f"{i}_{opp.player_name}")

    except Exception as e:
        logger.exception(e)
