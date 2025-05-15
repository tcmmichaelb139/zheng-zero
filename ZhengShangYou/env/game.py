class ZhengShangYou:
    def __init__(self, players) -> None:
        self.players = players
        self.current_player = 0
        self.current_pattern = None  # can also include the last played cards
