from ZhengShangYou.players.base_player import BasePlayer
from ZhengShangYou.env.utils import input2cards


class InputPlayer(BasePlayer):
    def __init__(self, player_id: int, player_name: str = None):
        super().__init__(player_id, player_name)

    def _play(self, info):
        valid_moves = self._get_valid_moves(info)

        while True:
            cards = self._get_input_cards()
            if cards in valid_moves:
                return cards
            else:
                print("Invalid move. Please try again.")
