from ZhengShangYou.env.env import Env
from ZhengShangYou.env.utils import _print_cards


class ZhengShangYou:
    def __init__(self, players, params) -> None:
        self.players = []
        for i in range(len(players)):
            self.players.append(DummyPlayer(i, players[i].player_name))

        self._env = Env(self.players, params)

        self._env.reset()

        self.params = params
        self._env._deal_cards()

        self.additional_rewards = [0.0] * len(self.players)

    def reset(self):
        self._env.reset()

        self._env._deal_cards()

        self.additional_rewards = [0.0] * len(self.players)

        return self._get_obs()

    def step(self, action):
        """
        Play a step of the game.
        """

        self.players[self._current_player()].set_action(action)
        self._env.step()

        obs = None
        reward = 0.0
        game_over = False

        # incentivized to play a card
        if action != []:
            reward += 0.25

        if self._game_over():
            game_over = True
        else:
            obs = self._get_obs(self._current_player())

        return (
            obs,
            reward,
            game_over,
        )

    def get_final_rewards(self):
        """
        Get the final rewards of the game.
        """
        rewards = self.additional_rewards.copy()

        for i in range(len(self.players)):
            if i in self._env.results:
                if self._env.results.index(i) == 0:
                    rewards[i] += 0.75

        return rewards

    def _get_obs(self, player_id=None):
        """
        Get the current observation of the game.
        """

        obs = self._env._get_info(player_id)

        return obs

    def _game_over(self):
        """
        Check if the game is over.
        """
        return self._env._game_over()

    def _current_player(self):
        """
        Get the current player.
        """
        return self._env.current_player

    def _print_hands(self):
        """
        Print the hands of all players.
        """
        for player in self.players:
            player._print_hand()


class DummyPlayer:
    def __init__(self, player_id: int, player_name: str = None):
        self.player_id = player_id
        self.player_name = player_name if player_name else f"Player {player_id}"
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
        """
        for card in move:
            self.cards.remove(card)

    def _print_hand(self):
        """
        Print the player's hand.
        """

        _print_cards(self.cards, player=f"{self.player_id} ({self.player_name})")
