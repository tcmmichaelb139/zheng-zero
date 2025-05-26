from ZhengShangYou.players.base_player import BasePlayer
from ZhengShangYou.env.utils import input2cards


class InputPlayer(BasePlayer):
    def __init__(self, player_id: int):
        super().__init__(player_id)

    def _play(self, info):
        return input2cards(
            f"{self.player_id}: Enter the cards to play (comma-separated): "
        )
