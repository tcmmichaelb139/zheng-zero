from ZhengShangYou.players.base_player import BasePlayer

import torch
import torch.nn as nn
import numpy as np
import random
from collections import deque


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


def train_model(lr, gamma, epsilon, epsilon_min, epsilon_decay):
    """
    Train the model with the given parameters.
    :param lr: Learning rate
    :param gamma: Discount factor
    :param epsilon: Exploration rate
    :param epsilon_min: Minimum exploration rate
    :param epsilon_decay: Decay rate for exploration
    """

    model_params = {
        "lr": lr,
        "gamma": gamma,
        "epsilon": epsilon,
        "epsilon_min": epsilon_min,
        "epsilon_decay": epsilon_decay,
    }

    player = ZhengZeroPlayer(player_id=0, model_params=model_params)


class ZhengZeroPlayer(BasePlayer):
    def __init__(
        self, player_id, model_path=None, model_params=None, buffer_size=10000
    ):
        self.player_id = player_id
        self.cards = []
        self.model = DQN(input_dim=100, output_dim=69)

        if model_path:
            self.model.load_state_dict(torch.load(model_path))
            self.model.eval()
        else:
            self.params = model_params
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.params["lr"]
            )

        self.replay_buffer = deque(maxlen=buffer_size)

    def _play(self, info):
        if np.random.rand() < self.params["epsilon"]:
            # Exploration: select a random move
            return self._random_move(info)

        q_values = self.model(self._state(info))

        return torch.argmax(q_values).item()

    def _state(self, info):
        # Convert the game state to a tensor
        state = np.zeros(100)
        # Fill in the state based on the game information
        # ...
        return torch.tensor(state, dtype=torch.float32)

    def remember(self, state, action, reward, next_state, done):
        # Store the experience in a replay buffer
        self.replay_buffer.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        if len(self.replay_buffer) < batch_size:
            return

        mini_batch = random.sample(self.replay_buffer, batch_size)
        for state, action, reward, next_state, done in mini_batch:
            state = torch.tensor(state, dtype=torch.float32)
            next_state = torch.tensor(next_state, dtype=torch.float32)

            target = reward
            if not done:
                target += (
                    self.params["gamma"] * torch.max(self.model(next_state)).item()
                )

            target_f = self.model(state)
            target_f[action] = target

            # Train the model
            self.optimizer.zero_grad()
            output = self.model(state)
            loss = nn.MSELoss()(output, target_f)
            loss.backward()
            self.optimizer.step()

            if self.params["epsilon"] > self.params["epsilon_min"]:
                self.params["epsilon"] *= self.params["epsilon_decay"]
