from ZhengShangYou.env.env import Env


class ZhengShangYou:
    def __init__(self, players, params) -> None:
        self.players = []
        for i in range(len(players)):
            self.players.append(DummyPlayer(i))

        self._env = Env(self.players, params)

        self._env.reset()

        self._env._deal_cards()

    def reset(self):
        self._env.reset()

        self._env._deal_cards()

        return self._get_obs()

    def step(self, action, add_info={}):
        """
        Play a step of the game.
        :return: The game information
        """

        current_player = self._current_player()
        results = self._env.results.copy()

        self.players[self._current_player()].set_action(action)
        self._env.step()

        obs = None
        reward = 0.0
        game_over = False

        if current_player in self._env.results and results != self._env.results:
            if self._env.results.index(current_player) == 0:
                reward += 5.0
            elif self._env.results.index(current_player) == 1:
                reward += -5.0
            elif self._env.results.index(current_player) == 2:
                reward += -5.0
            elif self._env.results.index(current_player) == 3:
                reward += -5.0

            for i in range(len(self.players)):
                if current_player == i:
                    continue

                reward += 0.1 * len(self.players[i].cards)

        if len(action) > 1:
            reward += 0.3 * len(action)
        if len(self._env.results) == 3 and self.players[current_player].cards != []:
            reward -= 10.0

        if self._game_over():
            game_over = True
        else:
            obs = self._get_obs(self._current_player())

        self._env.post_step(action)

        return (
            obs,
            reward,
            game_over,
            {"player_next_obs": self._get_obs(current_player)},
        )

    def _get_obs(self, player_id=None):
        """
        Get the current observation of the game.
        :return: The game information
        """

        obs = self._env._get_info(player_id)

        return obs

    def _game_over(self):
        """
        Check if the game is over.
        :return: True if the game is over, False otherwise
        """
        return self._env._game_over()

    def _current_player(self):
        """
        Get the current player.
        :return: The current player
        """
        return self._env.current_player


class DummyPlayer:
    def __init__(self, player_id: int):
        self.player_id = player_id
        self.cards = []
        self.action = None

    def reset(self):
        self.cards = []
        self.action = None

    def act(self):
        return self.action

    def set_action(self, action):
        self.action = action

    def _deal_hand(self, cards):
        """
        Deal cards to the player.
        :param cards: The cards to be dealt
        """
        self.cards = cards
        self._sort_hand()

    def _sort_hand(self):
        """
        Sort the player's hand.
        """
        self.cards.sort(key=lambda x: (x[0], x[1]))

    def _has_card(self, card):
        return card in self.cards

    def _played_hand(self, move):
        """
        Remove the played cards from the player's hand.
        :param move: The cards to be removed
        """
        for card in move:
            self.cards.remove(card)
