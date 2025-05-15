from ZhengShangYou.env.move_generator import MoveGenerator

import numpy as np


class BasePlayer:
    def __init__(self, player_id: int):
        self.player_id = player_id
        self.cards = []

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

    def play(self, info):
        """
        Play a card.
        :param info: The game information
        :return: The card to be played
        """

        selected_move = self._play(info)
        self._played_hand(selected_move)
        return selected_move

    def _play(self, info):
        """
        Play a card.
        :param info: The game information
        :return: The card to be played
        """
        raise NotImplementedError("This method should be overridden by subclasses")

    def _played_hand(self, move):
        """
        Remove the played cards from the player's hand.
        :param move: The cards to be removed
        """
        for card in move:
            self.cards.remove(card)

    def _random_move(self, info):
        # Randomly select a card from the player's hand

        trick = info["trick"]
        last_played_cards = info["last_played_cards"]

        move_gen = MoveGenerator(self.cards)

        moves = move_gen.generate_based_on_trick(trick, last_played_cards)

        valid_moves = []

        for move in moves:
            if move_gen._valid_move(
                trick,
                move,
                last_played_cards,
            ):
                valid_moves.append(move)

        assert len(valid_moves) > 0, "Should have at least one valid move"

        # select a random move from vlaid moves
        # weight the moves based on the number of cards

        selected_move = valid_moves[0]

        if len(valid_moves) == 2:
            selected_move = valid_moves[1]
        elif len(valid_moves) > 2:
            move_chances = [len(move) for move in valid_moves]

            move_chances = np.exp(move_chances) / np.sum(np.exp(move_chances))

            selected_move = valid_moves[
                np.random.choice(len(valid_moves), p=move_chances)
            ]

        return selected_move
