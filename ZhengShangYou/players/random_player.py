from ZhengShangYou.env.move_generator import MoveGenerator
from ZhengShangYou.players.base_player import BasePlayer
import numpy as np


class RandomPlayer(BasePlayer):
    def __init__(self, player_id: int, player_name: str = None):
        super().__init__(player_id, player_name)

    def _play(self, info):
        # Randomly select a card from the player's hand

        return self._random_move(info)
