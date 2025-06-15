from ZhengShangYou.players.base_player import BasePlayer
from ZhengShangYou.env.utils import input2cards, _print_card, _print_cards


class InputPlayer(BasePlayer):
    def __init__(self, player_id: int, player_name: str = None):
        super().__init__(player_id, player_name)

    def _play(self, info):
        valid_moves = self._get_valid_moves(info)

        if len(valid_moves) == 1 and valid_moves[0] == []:
            print(f"{self.player_name} ({self.player_id}) has no valid moves.")
            return []

        print(f"Last played cards: ", end="")
        if not info["last_played_cards"]:
            print("None")
        else:
            for card in info["last_played_cards"]:
                _print_card(card, end=" ")
            print()

        print("Current cards in hand:")
        for i, card in enumerate(info["cards"]):
            print(i, end=": ")
            _print_card(card)

        while True:
            cards = self._get_input_cards(info["cards"])
            if cards in valid_moves:
                return cards
            else:
                print("Invalid move. Please try again.")

    def _get_input_cards(self, cards):
        return input2cards(f"{self.player_name} ({self.player_id})'s turn: ", cards)
