from ZhengShangYou.env.zhengshangyou import ZhengShangYou
from ZhengShangYou.players.base_player import BasePlayer
from ZhengShangYou.env.move_generator import MoveGenerator
from ZhengShangYou.env.utils import (
    card2int,
    int2card,
    trick2int,
    int2trick,
    create_logger,
)
from ZhengShangYou.zhengzero.sumtree import SumTree

import os
import torch
import torch.nn as nn
import numpy as np
import random
from collections import deque
from copy import deepcopy
import matplotlib.pyplot as plt


logger = create_logger(__name__)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 512)
        self.fc5 = nn.Linear(512, 512)
        self.fc6 = nn.Linear(512, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = self.fc6(x)
        return x


def train_model(
    players,
    zhengzero_player_ids,
    batch_size,
    params,
):
    """
    Train the model with the given parameters.
    :param players: The players to be trained
    :param batch_size: The batch size
    :param num_episodes: The number of episodes to train
    """

    episode = 0
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
        # plot the reward

        plt.scatter(
            range(len(all_rewards[0])),
            all_rewards[0],
            s=1,
            c="blue",
            alpha=0.5,
            label="Reward",
        )
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Reward vs Episode")
        plt.savefig("reward.png")
        plt.show()
    except Exception as e:
        logger.exception(e)


class ReplayBuffer:
    def __init__(self, buffer_size, alpha=0.1, beta=0.1, epsilon=0.01):
        self.sumtree = SumTree(buffer_size)
        self.buffer_size = buffer_size
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = 0.001
        self.epsilon = epsilon

    def add(self, experience):
        max_p = self.sumtree.get_max
        self.sumtree.add(max_p, experience)

    def sample(self, batch_size):
        batch = []
        idxs = []
        segment = self.sumtree.total / batch_size
        priorities = []

        self.beta = np.min([1.0, self.beta + self.beta_increment])

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.sumtree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = priorities / self.sumtree.total
        is_weight = np.power(self.sumtree.size * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        return batch, idxs, is_weight

    def update(self, idx, td_errors):
        new_priorities = (np.abs(td_errors) + self.epsilon) ** self.alpha

        self.sumtree.update(idx, new_priorities)

    def __len__(self):
        return self.sumtree.size


class ZhengZeroPlayer(BasePlayer):
    def __init__(
        self, player_id, model_path=None, model_params=None, buffer_size=10000
    ):
        super().__init__(player_id)
        self.model = DQN(input_dim=224, output_dim=1).to(device)
        self.target_model = DQN(input_dim=224, output_dim=1).to(device)

        self.model_path = model_path
        self.load_model()

        self.params = model_params
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.params["lr"],
        )

        self.model.eval()

        self.consec_passes = 0

        self.replay_buffer = ReplayBuffer(
            buffer_size, alpha=0.6, beta=0.4, epsilon=0.00001
        )
        self.buffer_size = buffer_size
        self.current_replay = []

    def _play(self, info):
        valid_moves = self._get_valid_moves(info)

        if np.random.rand() <= self.params["epsilon"]:
            return self._random_move(info)

        state_moves = torch.tensor(
            np.array([state2array(info, move) for move in valid_moves]),
            dtype=torch.float32,
        ).to(device)

        with torch.no_grad():
            q_value = self.model(state_moves).squeeze(1).cpu().numpy()

        selected_move = valid_moves[np.argmax(q_value)]

        return selected_move

    def _get_valid_moves(self, info):
        """
        Get the valid moves for the current player.
        """
        trick = info["trick"]
        last_played_cards = info["last_played_cards"]

        move_gen = MoveGenerator(info["cards"])

        moves = move_gen.generate_based_on_trick(trick, last_played_cards)

        valid_moves = []

        for move in moves:
            if move_gen._valid_move(
                trick,
                move,
                last_played_cards,
            ):
                valid_moves.append(move)

        return valid_moves

    def remember(self, state, action, reward, done):
        if not done:
            self.current_replay.append((deepcopy(state), action, reward, done))
        if done:
            if self.params["train"]:
                total_reward = -10

                for i, (s, a, r, d) in enumerate(self.current_replay):
                    total_reward = max(total_reward, r)
                    if i == len(self.current_replay) - 1:
                        break
                    self.replay_buffer.add(
                        (
                            state2array(s, a),
                            a,
                            reward + r,
                            deepcopy(self.current_replay[i + 1][0]),
                        )
                    )

                total_reward += reward

                logger.info(f"total reward: {total_reward:0.2f}")
            self.current_replay = []

    def update_target_model(self):
        # Update the target model with the current model's weights
        self.target_model.load_state_dict(self.model.state_dict())

    def save_model(self):
        if self.model_path is None:
            logger.info("Model path not specified. Model not saved.")
            return

        torch.save(self.model.state_dict(), self.model_path)

    def load_model(self):
        if self.model_path:
            if os.path.exists(self.model_path):
                self.model.load_state_dict(torch.load(self.model_path))
                self.target_model.load_state_dict(torch.load(self.model_path))

    def replay(self, batch_size, episode):
        if self.params["train"] is False:
            return
        if len(self.replay_buffer) < batch_size * self.params["replay_num"] * 2:
            return

        total_loss = 0

        for _ in range(self.params["replay_num"]):
            mini_batch, indices, is_weights = self.replay_buffer.sample(batch_size)
            is_weights = torch.tensor(is_weights, dtype=torch.float32).to(device)

            batch_state, _, batch_reward, batch_next_state_full = zip(*mini_batch)

            batch_state = np.array([s for s, _, _, _ in mini_batch])
            batch_state = torch.tensor(batch_state, dtype=torch.float32).to(device)
            batch_reward = (
                torch.tensor(batch_reward, dtype=torch.float32).to(device).unsqueeze(1)
            )
            batch_next_state = torch.tensor(
                [[0.0]] * len(batch_state), dtype=torch.float32
            ).to(device)

            for i, next_state in enumerate(batch_next_state_full):
                valid_moves = self._get_valid_moves(next_state)

                next_state_tensor = np.array(
                    [state2array(next_state, move) for move in valid_moves]
                )

                next_state_tensor = torch.tensor(
                    next_state_tensor, dtype=torch.float32
                ).to(device)

                with torch.no_grad():
                    q_values = self.target_model(next_state_tensor)

                batch_next_state[i] = torch.max(q_values.squeeze(1))

            predicted = self.model(batch_state)
            target = batch_reward + self.params["gamma"] * batch_next_state

            td_error = torch.abs(predicted - target).cpu().detach().numpy()

            for i in range(batch_size):
                self.replay_buffer.update(indices[i], td_error[i])

            loss = (is_weights * (predicted - target) ** 2).mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        if self.params["epsilon"] > self.params["epsilon_min"]:
            self.params["epsilon"] *= self.params["epsilon_decay"]

        if episode % self.params["update_length"] == 0:
            logger.info("model updated")
            self.update_target_model()
            if self.params["save_model"]:
                self.save_model()


def state2array(info, move):
    """
    Convert the game information to a numpy array.
    :param info: The game information
    :return: The game information as a numpy array
    """

    # "trick": self.current_trick,
    # "last_played_cards": self.last_played_cards,
    # "round_passed": self.round_passed,
    # "rounds": self.rounds,
    # "history": self.history,
    # "cards": self.players[self.current_player].cards,

    state = np.zeros(224)

    idx = 0

    state[trick2int(info["trick"])] = 1

    idx += 8

    if info["last_played_cards"] is not None:
        for card in info["last_played_cards"]:
            state[idx + card2int(card)] = 1

    idx += 54

    for i, card in enumerate(info["cards_played"]):
        state[idx + i] = card

    idx += 54

    for card in info["cards"]:
        state[idx + card2int(card)] = 1

    idx += 54

    for card in move:
        state[idx + card2int(card)] = 1

    return state


def array2state(arr):
    """
    Convert the numpy array to game information.
    :param arr: The numpy array to be converted
    :return: The game information
    """
    trick = arr[:8]
    last_played_cards = arr[8:62]
    cards_played = arr[62:116]
    cards = arr[116:170]
    action = arr[170:224]

    trick = int2trick(np.argmax(trick))
    last_played_cards = array2cards(last_played_cards)
    cards_played = array2cards(cards_played)
    cards = array2cards(cards)
    action = array2cards(action)

    return (
        {
            "trick": trick,
            "last_played_cards": last_played_cards,
            "cards_played": cards_played,
            "cards": cards,
        },
        action,
    )


def cards2array(cards):
    """
    Convert the cards to a numpy array.
    :param cards: The cards to be converted
    :return: The cards as a numpy array
    """
    arr = np.zeros(54)
    for card in cards:
        arr[card2int(card)] = 1
    return arr


def array2cards(arr):
    """
    Convert the numpy array to cards.
    :param arr: The numpy array to be converted
    :return: The cards
    """
    cards = []
    for i in range(len(arr)):
        if arr[i] == 1:
            cards.append(int2card(i))
    return cards
