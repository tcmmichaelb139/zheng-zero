import numpy as np

from ZhengShangYou.env.move_detector import detect_move
from ZhengShangYou.env.utils import _print_cards, card2int

DECK = [
    ("3", "Spades"),
    ("3", "Hearts"),
    ("3", "Clubs"),
    ("3", "Diamonds"),
    ("4", "Spades"),
    ("4", "Hearts"),
    ("4", "Clubs"),
    ("4", "Diamonds"),
    ("5", "Spades"),
    ("5", "Hearts"),
    ("5", "Clubs"),
    ("5", "Diamonds"),
    ("6", "Spades"),
    ("6", "Hearts"),
    ("6", "Clubs"),
    ("6", "Diamonds"),
    ("7", "Spades"),
    ("7", "Hearts"),
    ("7", "Clubs"),
    ("7", "Diamonds"),
    ("8", "Spades"),
    ("8", "Hearts"),
    ("8", "Clubs"),
    ("8", "Diamonds"),
    ("9", "Spades"),
    ("9", "Hearts"),
    ("9", "Clubs"),
    ("9", "Diamonds"),
    ("10", "Spades"),
    ("10", "Hearts"),
    ("10", "Clubs"),
    ("10", "Diamonds"),
    ("J", "Spades"),
    ("J", "Hearts"),
    ("J", "Clubs"),
    ("J", "Diamonds"),
    ("Q", "Spades"),
    ("Q", "Hearts"),
    ("Q", "Clubs"),
    ("Q", "Diamonds"),
    ("K", "Spades"),
    ("K", "Hearts"),
    ("K", "Clubs"),
    ("K", "Diamonds"),
    ("A", "Spades"),
    ("A", "Hearts"),
    ("A", "Clubs"),
    ("A", "Diamonds"),
    ("2", "Spades"),
    ("2", "Hearts"),
    ("2", "Clubs"),
    ("2", "Diamonds"),
    ("LJ", "Spades"),  # low joker
    ("HJ", "Spades"),  # high joker
]
CARDS = {
    "3": 0,
    "4": 1,
    "5": 2,
    "6": 3,
    "7": 4,
    "8": 5,
    "9": 6,
    "10": 7,
    "J": 8,
    "Q": 9,
    "K": 10,
    "A": 11,
    "2": 12,
    "LJ": 13,  # low joker
    "HJ": 14,
}
SUITS = {
    "Spades": 0,  # Spades
    "Hearts": 1,  # Hearts
    "Clubs": 2,  # Clubs
    "Diamonds": 3,  # Diamonds
}


class Env:
    def __init__(self, players) -> None:
        self.players = players
        self.current_player = 0
        self.trick_leader = 0  # the player who (currently) leads the trick
        self.current_trick = None  # the current trick
        self.last_played_cards = None  # the last played cards
        self.round_passed = np.array(
            [False] * len(players)
        )  # whether the player has passed in this round

        self.rounds = 0  # number of rounds played
        self.results = []  # the results of the game

        self.cards_played = [0] * 54
        self.history = []

    def reset(self):
        for player in self.players:
            player.reset()

        self.current_player = 0
        self.trick_leader = 0
        self.current_trick = None
        self.last_played_cards = None
        self.round_passed = np.array([False] * len(self.players))

        self.rounds = 0
        self.results = []

        self.cards_played = [0] * 54
        self.history = []

    def step(self):
        """
        Play a step of the game.
        :return: The game information
        """

        move = self.players[self.current_player].act()

        for m in move:
            self.cards_played[card2int(m)] = 1

        # _print_cards(move, self.players[self.current_player].player_id)

        if move != []:
            self.trick = detect_move(move)
            self.last_played_cards = move
            self.round_passed[self.current_player] = False
            self.trick_leader = self.current_player
            self.current_trick = self.trick
        else:
            self.round_passed[self.current_player] = True

        self.history.append(move)

        if not self._is_player_finished():
            self._players_played_out()

        self._next_player()

        if self._is_round_over():
            self._new_round()

    def _game_over(self):
        """
        Check if the game is over.
        :return: True if the game is over, False otherwise
        """
        return len(self.results) == len(self.players)

    def _deal_cards(self):
        """
        Deal cards to players.
        """

        deck = DECK.copy()
        for i in range(len(deck)):
            deck[i] = (CARDS[deck[i][0]], SUITS[deck[i][1]])
        np.random.shuffle(deck)
        for i in range(len(self.players)):
            hand = deck[i * 13 : (i + 1) * 13]
            if i <= 1:
                hand.extend([deck[-2 + i]])
            self.players[i]._deal_hand(hand)

    def _get_info(self):
        """
        Get the game information.
        :return: The game information
        """
        info = {
            "trick": self.current_trick,
            "last_played_cards": self.last_played_cards,
            "round_passed": self.round_passed,
            "rounds": self.rounds,
            "history": self.history,
            "cards_played": self.cards_played,
            "cards": self.players[self.current_player].cards,
        }
        return info

    def _is_round_over(self):
        """
        Check if the round is over.
        :return: True if the round is over, False otherwise
        """

        return (
            np.sum(self.round_passed) == len(self.players) - 1
            and self.current_player == self.trick_leader
        )

    def _new_round(self):
        """
        Start a new round.
        """
        self.rounds += 1
        self.round_passed = np.array([False] * len(self.players))
        self.current_trick = None
        self.last_played_cards = None

    def _next_player(self):
        """
        Move to the next player.
        """
        self.current_player = (self.current_player + 1) % len(self.players)

    def _players_played_out(self):
        """
        Check if the player has played out.
        :return: True if the player has played out, False otherwise
        """
        if len(self.players[self.current_player].cards) == 0:
            self.results.append(self.current_player)

    def _is_player_finished(self):
        """
        Check if the player has finished.
        :return: True if the player has finished, False otherwise
        """
        return self.results.count(self.current_player) > 0
