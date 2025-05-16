from ZhengShangYou.env.zhengshangyou import ZhengShangYou
from ZhengShangYou.players.base_player import BasePlayer
from ZhengShangYou.env.move_generator import MoveGenerator
from ZhengShangYou.env.utils import card2int, int2card, trick2int, int2trick

import torch
import torch.nn as nn
import numpy as np
import random
from collections import deque
from tqdm import tqdm


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
    num_episodes,
):
    """
    Train the model with the given parameters.
    :param players: The players to be trained
    :param batch_size: The batch size
    :param num_episodes: The number of episodes to train
    """

    env = ZhengShangYou(players)

    win_count = [0] * len(players)

    for episode in range(num_episodes):
        print(f"##### Episode {episode + 1}/{num_episodes} #####")

        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            current_player = env._current_player()
            action = players[current_player].play(state)
            next_state, reward, done, _ = env.step(action)

            players[current_player].remember(state, action, reward, next_state, done)
            state = next_state

            total_reward += reward

        for i in zhengzero_player_ids:
            players[i].replay(batch_size, episode)

        win_count[env._env.results[0]] += 1
        if episode % 100 == 0:
            print(f"Win count: {win_count}")


class ZhengZeroPlayer(BasePlayer):
    def __init__(
        self, player_id, model_path=None, model_params=None, buffer_size=10000
    ):
        super().__init__(player_id)
        self.model = DQN(input_dim=224, output_dim=1).to(device)
        self.target_model = DQN(input_dim=224, output_dim=1).to(device)

        if model_path:
            self.model.load_state_dict(torch.load(model_path))
            self.model.eval()
        else:
            self.params = model_params
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.params["lr"]
            )

        self.replay_buffer = deque(maxlen=buffer_size)
        self.current_replay = []

    def _play(self, info):
        if np.random.rand() < self.params["epsilon"]:
            # Exploration: select a random move
            return self._random_move(info)

        valid_moves = self._get_valid_moves(info)

        q_value = [0] * len(valid_moves)

        for i, move in enumerate(valid_moves):
            state = state2array(info, move)
            state = torch.tensor(state, dtype=torch.float32).to(device)
            with torch.no_grad():
                q_value[i] = self.model(state).item()

        valid_moves.reverse()
        q_value.reverse()

        return valid_moves[np.argmax(q_value)]

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

        self.current_replay.append((state, action, reward, next_state, done))
        if reward != 0:
            for s, a, _, ns, d in self.current_replay:
                self.replay_buffer.append((state2array(s, a), a, reward, ns, d))
            self.current_replay = []

    def update_target_model(self):
        # Update the target model with the current model's weights
        self.target_model.load_state_dict(self.model.state_dict())

    def replay(self, batch_size, episode):
        if len(self.replay_buffer) < batch_size:
            return
        if episode % batch_size == 0:
            self.update_target_model()

        mini_batch = random.sample(self.replay_buffer, batch_size)

        for state, action, reward, next_state, done in mini_batch:
            if reward == 0:
                continue
            state = torch.tensor(state, dtype=torch.float32).to(device)
            reward_tensor = torch.tensor([reward], dtype=torch.float32).to(device)

            next_q_value = torch.tensor([0.0], dtype=torch.float32).to(device)

            if not done:
                valid_moves = self._get_valid_moves(next_state)

                for i, move in enumerate(valid_moves):
                    next_state_tensor = state2array(next_state, move)
                    next_state_tensor = torch.tensor(
                        next_state_tensor, dtype=torch.float32
                    ).to(device)

                    with torch.no_grad():
                        q_value = self.target_model(next_state_tensor).item()

                    if q_value > next_q_value.item():
                        next_q_value = torch.tensor([q_value], dtype=torch.float32).to(
                            device
                        )

            target = reward_tensor + self.params["gamma"] * next_q_value
            predicted = self.model(state).squeeze()

            loss = nn.MSELoss()(predicted, target.squeeze())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

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

    for card in info["cards_played"]:
        state[idx + card] = 1

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
    cards = arr[124:]

    trick = int2trick(np.argmax(trick))
    last_played_cards = array2cards(last_played_cards)
    cards_played = array2cards(cards_played)
    cards = array2cards(cards)

    return {
        "trick": trick,
        "last_played_cards": last_played_cards,
        "cards_played": cards_played,
        "cards": cards,
    }


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
