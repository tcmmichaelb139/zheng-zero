from ZhengShangYou.env.zhengshangyou import ZhengShangYou
from ZhengShangYou.players.base_player import BasePlayer
from ZhengShangYou.env.move_generator import MoveGenerator
from ZhengShangYou.env.utils import card2int, int2card, trick2int, int2trick

import os
import torch
import torch.nn as nn
import numpy as np
import random
from collections import deque
from tqdm import tqdm
from copy import deepcopy


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
    replay_num,
    num_episodes,
    params,
):
    """
    Train the model with the given parameters.
    :param players: The players to be trained
    :param batch_size: The batch size
    :param num_episodes: The number of episodes to train
    """

    env = ZhengShangYou(players, params)

    win_count = np.zeros((len(players), len(players)), dtype=int)

    for episode in range(num_episodes):
        print(f"##### Episode {episode + 1}/{num_episodes} #####")

        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            current_player = env._current_player()
            action, add_info = players[current_player].play(state)
            next_state, reward, done, addit = env.step(action, add_info)

            players[current_player].remember(
                state, action, reward, addit["player_next_obs"], done
            )
            state = next_state

            total_reward += reward

        if params["train"]:
            for i in zhengzero_player_ids:
                players[i].replay(batch_size, replay_num, episode)

        for i, j in enumerate(env._env.results):
            win_count[j][i] += 1

        if episode % 100 == 0:
            # print wins
            for i in range(len(players)):
                print(
                    f"Player {i} wins: {win_count[i]}",
                )
            win_count = np.zeros((len(players), len(players)), dtype=int)


class ZhengZeroPlayer(BasePlayer):
    def __init__(
        self, player_id, model_path=None, model_params=None, buffer_size=10000
    ):
        super().__init__(player_id)
        self.model = DQN(input_dim=224, output_dim=1).to(device)
        self.target_model = DQN(input_dim=224, output_dim=1).to(device)

        self.model_path = model_path

        if model_path:
            if os.path.exists(model_path):
                self.model.load_state_dict(torch.load(model_path))
                self.target_model.load_state_dict(torch.load(model_path))

        self.params = model_params
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params["lr"])

        self.model.eval()

        self.consec_passes = 0

        self.replay_buffer = deque(maxlen=buffer_size)
        self.buffer_size = buffer_size
        self.current_replay = []

    def _play(self, info):
        valid_moves = self._get_valid_moves(info)

        if np.random.rand() <= self.params["epsilon"]:
            # Exploration: select a random move
            return self._random_move(info), {"valid_moves": valid_moves}

        state_moves = torch.tensor(
            np.array([state2array(info, move) for move in valid_moves]),
            dtype=torch.float32,
        ).to(device)

        with torch.no_grad():
            q_value = self.model(state_moves).squeeze(1).cpu().numpy()

        selected_move = valid_moves[np.argmax(q_value)]

        # print(selected_move, q_value)

        # print(selected_move, len(valid_moves))

        if selected_move == [] and len(valid_moves) > 1:
            self.consec_passes += 1
        else:
            self.consec_passes = 0

        return selected_move, {
            "valid_moves": valid_moves,
            "consec_passes": self.consec_passes,
        }

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

    def remember(self, state, action, reward, next_state, done):
        # Store the experience in a replay buffer
        if len(self.current_replay) == 0 and state["cards"] == []:
            return

        self.current_replay.append(
            (deepcopy(state), action, reward, deepcopy(next_state), done)
        )
        if len(next_state["cards"]) == 0 or done:
            # only store good experiences
            total_reward = 0
            # if len(self.current_replay) <= 30:
            for i, (s, a, r, ns, d) in enumerate(self.current_replay):
                total_reward += r
                self.replay_buffer.append(
                    (
                        state2array(s, a),
                        a,
                        reward + r,
                        (
                            deepcopy(
                                self.current_replay[i + 1][0]
                                if i + 1 < len(self.current_replay)
                                else None
                            )
                        ),
                        len(ns["cards"]) == 0,
                    )
                )
            print("total reward", total_reward)
            self.current_replay = []

    def update_target_model(self):
        # Update the target model with the current model's weights
        self.target_model.load_state_dict(self.model.state_dict())

    def save_model(self):
        if self.model_path is None:
            print("Model path not specified. Model not saved.")
            return

        torch.save(self.model.state_dict(), self.model_path)

    def replay(self, batch_size, replay_num, episode):
        if len(self.replay_buffer) < batch_size * replay_num * 2:
            return
        if episode % 100 == 0:
            print("model updated")
            self.update_target_model()
            self.save_model()

        total_loss = 0

        for _ in range(replay_num):
            mini_batch = random.sample(self.replay_buffer, batch_size)

            batch_state, _, batch_reward, batch_next_state_full, done = zip(*mini_batch)

            batch_state = np.array([s for s, _, _, _, _ in mini_batch])
            batch_state = torch.tensor(batch_state, dtype=torch.float32).to(device)
            batch_reward = (
                torch.tensor(batch_reward, dtype=torch.float32).to(device).unsqueeze(1)
            )
            batch_next_state = torch.tensor(
                [[0.0]] * len(batch_state), dtype=torch.float32
            ).to(device)

            for i, next_state in enumerate(batch_next_state_full):
                if not done[i]:
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

            loss = nn.MSELoss()(predicted, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        if self.params["epsilon"] > self.params["epsilon_min"]:
            self.params["epsilon"] *= self.params["epsilon_decay"]


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
