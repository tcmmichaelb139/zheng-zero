from vars import GLOBAL_LOG_FOLDER
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
from copy import deepcopy

from trueskill import Rating


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


class ReplayBuffer:
    def __init__(self, buffer_size, alpha=0.6, beta=0.4, epsilon=1e-6):
        self.sumtree = SumTree(buffer_size)
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = (1.0 - beta) / buffer_size
        self.epsilon = epsilon

    def _get_priority(self, error):
        return (np.abs(error) + self.epsilon) ** self.alpha

    def add(self, experience):
        max_p = self.sumtree.get_max
        self.sumtree.add(max_p, experience)

    def sample(self, batch_size):
        batch = []
        indices = []
        priorities = []
        segment = self.sumtree.total / batch_size

        self.beta = np.min([1.0, self.beta + self.beta_increment])

        for i in range(batch_size):
            a = i * segment
            b = (i + 1) * segment
            a = b

            s = random.uniform(a, b)

            idx, p, data = self.sumtree.get(s)
            priorities.append(p)
            batch.append(data)
            indices.append(idx)

        sampling_probabilities = np.array(priorities) / self.sumtree.total
        weights = np.power(
            self.sumtree.size * sampling_probabilities + self.epsilon,
            -self.beta,
        )
        weights /= weights.max()

        return batch, indices, weights

    def update(self, idx, error):
        p = self._get_priority(error)
        self.sumtree.update(idx, p)

    def __len__(self):
        return self.sumtree.size


class ZhengZeroPlayer(BasePlayer):
    def __init__(
        self,
        player_id: int,
        player_name: str = None,
        model_path=None,
        model_params=None,
        buffer_size=10000,
    ):
        super().__init__(player_id, player_name)
        self.model = DQN(input_dim=224, output_dim=1).to(device)
        self.target_model = DQN(input_dim=224, output_dim=1).to(device)

        self.model_path = model_path
        if model_params["save_model"]:
            self.model_dir = GLOBAL_LOG_FOLDER + f"/models/{player_id}/"
            if os.path.exists(self.model_dir) is False:
                os.makedirs(self.model_dir)
        self.load_model()

        self.params = model_params
        if self.params["train"]:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.params["lr"],
            )

        if self.params["train"]:
            self.replay_buffer = ReplayBuffer(buffer_size)
            self.buffer_size = buffer_size
            self.current_replay = []

        self.rating = Rating()

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

    def remember(self, state, action, reward, done):
        if self.params["train"] is False:
            return
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

    def save_model(self, episode=None):
        torch.save(
            self.model.state_dict(),
            self.model_dir + f"zheng-zero-{episode if episode is not None else ''}.pth",
        )

    def load_model(self):
        if self.model_path:
            if os.path.exists(self.model_path):
                self.model.load_state_dict(
                    torch.load(self.model_path, weights_only=True)
                )
                self.target_model.load_state_dict(
                    torch.load(self.model_path, weights_only=True)
                )

        self.model.eval()

    def replay(self, batch_size, episode):
        if self.params["train"] is False:
            return
        if len(self.replay_buffer) < batch_size * self.params["replay_num"] * 2:
            return

        total_loss = 0

        for _ in range(self.params["replay_num"]):
            mini_batch, indices, weights = self.replay_buffer.sample(batch_size)
            weights = torch.tensor(weights, dtype=torch.float32).to(device)

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

            for i, idx in enumerate(indices):
                self.replay_buffer.update(idx, td_error[i])

            loss = (weights * (predicted - target) ** 2).mean()
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
                self.save_model(episode)

    def clone(self, episode=None):
        """
        Create a clone of the current player without copying replay buffer.
        :return: A new instance of ZhengZeroPlayer with the same parameters
        """
        player = deepcopy(
            ZhengZeroPlayer(
                player_id=self.player_id,
                player_name=(
                    f"Player-{episode}" if episode is not None else self.player_name
                ),
                model_path=None,
                model_params=self.params,
                buffer_size=self.buffer_size,
            )
        )

        player.model.load_state_dict(self.model.state_dict())
        player.target_model.load_state_dict(self.target_model.state_dict())
        player.rating = deepcopy(self.rating)

        return player


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
