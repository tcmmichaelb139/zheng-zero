import itertools
import collections
import numpy as np

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


class MoveGenerator:
    def __init__(self, cards_list) -> None:
        self.cards_list = cards_list
        self.cards_dict = collections.defaultdict(int)
        self.cards_value_dict = [[] for _ in range(len(CARDS))]

        for card in cards_list:
            self.cards_dict[card] += 1
            self.cards_value_dict[card[0]].append(card)

        self.single_card_moves = []
        self.pair_card_moves = []
        self.triple_card_moves = []
        self.bomb_card_moves = []

    def generate_single_card_moves(self):
        """
        Generate all possible single card moves.
        :return: A list of possible single card moves
        """
        self.single_card_moves = []
        for card in self.cards_list:
            self.single_card_moves.append([card])
        return self.single_card_moves

    def generate_pair_card_moves(self):
        """
        Generate all possible pair card moves.
        :return: A list of possible pair card moves
        """
        self.pair_card_moves = []
        for i in range(0, len(CARDS) - 1):
            if len(self.cards_value_dict[i]) >= 2:
                for combination in itertools.combinations(self.cards_value_dict[i], 2):
                    self.pair_card_moves.append(list(combination))

        return self.pair_card_moves

    def generate_triple_card_moves(self):
        """
        Generate all possible triple card moves.
        :return: A list of possible triple card moves
        """
        self.triple_card_moves = []
        for i in range(0, len(CARDS) - 1):
            if len(self.cards_value_dict[i]) >= 3:
                for combination in itertools.combinations(self.cards_value_dict[i], 2):
                    self.triple_card_moves.append(list(combination))
        return self.triple_card_moves

    def generate_bomb_card_moves(self):
        """
        Generate all possible bomb card moves.
        :return: A list of possible bomb card moves
        """
        self.bomb_card_moves = []
        for i in range(0, len(CARDS) - 1):
            if len(self.cards_value_dict[i]) >= 4:
                for combination in itertools.combinations(self.cards_value_dict[i], 4):
                    self.bomb_card_moves.append(list(combination))
        return self.bomb_card_moves

    def _generate_straight_moves(self, length, repeats=1):
        """
        Generate all possible straight moves.
        :param length: The length of the straight
        :param repeats: The number of repeats for each card
        :param flush: Whether to include flush moves
        :return: A list of possible straight moves
        """
        moves = []

        for start_card in range(0, len(CARDS) - length + 1):
            if len(self.cards_value_dict[start_card]) < repeats:
                continue
            straight_possibilities = [start_card]

            for i in range(1, length):
                next_card_value = start_card + i

                if next_card_value > 11:
                    break

                if len(self.cards_value_dict[next_card_value]) >= repeats:
                    straight_possibilities.append(next_card_value)

            if len(straight_possibilities) != length:
                continue

            set_possibilities = []
            for card in straight_possibilities:
                card_possibilities = []
                for i in range(4):
                    if self.cards_dict[(card, i)]:
                        card_possibilities.append((card, i))
                set_possibilities.append(card_possibilities)

            selections_per_value = [
                itertools.product(options, repeat=repeats)
                for options in set_possibilities
            ]

            diff_suit_possibilities = []
            same_suit_possibilities = []
            for card_set in itertools.product(*selections_per_value):
                potential_straight = list(itertools.chain(*card_set))

                is_sorted = all(
                    potential_straight[i] <= potential_straight[i + 1]
                    for i in range(len(potential_straight) - 1)
                )

                if potential_straight[-1][0] >= 12:
                    continue

                if is_sorted and len(potential_straight) == len(
                    set(tuple(card) for card in potential_straight)
                ):
                    if len(set([card[0][1] for card in card_set])) == 1:
                        same_suit_possibilities.append(potential_straight)
                    else:
                        diff_suit_possibilities.append(potential_straight)

            # TODO check if need to sort
            # diff_suit_possibilities.sort()
            # same_suit_possibilities.sort()

            moves.extend(diff_suit_possibilities)
            moves.extend(same_suit_possibilities)

        return moves

    def generate_straight_single_moves(self, length):
        """
        Generate all possible straight moves.
        :return: A list of possible straight moves
        """
        return self._generate_straight_moves(length, 1)

    def generate_straight_pair_moves(self, length):
        """
        Generate all possible straight moves.
        :return: A list of possible straight moves
        """
        return self._generate_straight_moves(length, 2)

    def generate_straight_triple_moves(self, length):
        """
        Generate all possible straight moves.
        :return: A list of possible straight moves
        """
        return self._generate_straight_moves(length, 3)

    def generate_all_moves(self):
        """
        Generate all possible straight moves.
        :return: A list of possible straight moves
        """
        moves = [[]]
        moves.extend(self.generate_single_card_moves())
        moves.extend(self.generate_pair_card_moves())
        moves.extend(self.generate_triple_card_moves())
        moves.extend(self.generate_bomb_card_moves())
        for length in range(3, 13):
            for repeats in range(1, 4):
                moves.extend(self._generate_straight_moves(length, repeats))

        return moves

    def generate_based_on_trick(self, trick, last_played_cards):
        """
        Generate all possible moves based on the current trick.
        :param trick: The current trick
        :return: A list of possible moves
        """

        if trick is None or trick == "pass":
            return self.generate_all_moves()

        l = len(last_played_cards)

        moves = [[]]

        if trick == "single":
            moves.extend(self.generate_single_card_moves())
        elif trick == "pair":
            moves.extend(self.generate_pair_card_moves())
        elif trick == "triple":
            moves.extend(self.generate_triple_card_moves())
        elif trick == "straight_1":
            assert l != -1
            moves.extend(self.generate_straight_single_moves(l))
        elif trick == "straight_22":
            assert l != -1
            moves.extend(self.generate_straight_pair_moves(l))
        elif trick == "straight_333":
            assert l != -1
            moves.extend(self.generate_straight_triple_moves(l))
        elif trick != "bomb":
            raise ValueError("Invalid trick type")

        moves.extend(self.generate_bomb_card_moves())

        return moves

    def _valid_move(self, trick, move, last_played_cards):
        """
        Check if the move is valid based on the current trick and last played cards.
        :param trick: The current trick
        :param move: The move to be checked
        :param last_played_cards: The last played cards
        :return: True if the move is valid, False otherwise
        """

        # TODO make sure straights end before 2

        if trick == None:
            return True

        if len(move) == 0:
            return True

        if trick == "bomb":
            return _is_bomb(move) and move[0][0] > last_played_cards[0][0]

        if _is_bomb(move):
            return True

        if trick == "single" or trick == "pair" or trick == "triple":
            return move[0][0] > last_played_cards[0][0]
        elif trick == "straight_1":
            if len(move) != len(last_played_cards):
                return False
            if move[-1][0] >= 12:
                return False
            return _is_higher_straight(move, last_played_cards, 1)
        elif trick == "straight_22":
            if len(move) != len(last_played_cards):
                return False
            if move[-1][0] >= 12:
                return False
            return _is_higher_straight(move, last_played_cards, 2)
        elif trick == "straight_333":
            if len(move) != len(last_played_cards):
                return False
            if move[-1][0] >= 12:
                return False
            return _is_higher_straight(move, last_played_cards, 3)
        else:
            raise ValueError("Invalid trick type")


def _is_higher_straight(move, last_played_cards, rep=1):
    if _is_straight(move, rep) and _is_straight(last_played_cards, rep):
        if _is_flush(move) and _is_flush(last_played_cards):
            return move[0][0] > last_played_cards[0][0]
        elif _is_flush(move) and not _is_flush(last_played_cards):
            return True
        elif not _is_flush(move) and _is_flush(last_played_cards):
            return False
        else:
            return move[0][0] > last_played_cards[0][0]

    return False


def _is_bomb(move):
    """
    Check if the move is a bomb.
    :param move: The move to be checked
    :return: True if the move is a bomb, False otherwise
    """
    return len(move) == 4 and move[0][0] == move[1][0] == move[2][0] == move[3][0]


def _is_straight(move, rep=1):
    """
    Check if the move is a straight.
    :param move: The move to be checked
    :return: True if the move is a straight, False otherwise
    """
    if len(move) < 3:
        return False
    if len(move) % rep != 0:
        return False

    for i in range(0, len(move), rep):
        for j in range(i, i + rep):
            if move[i][0] != move[j][0]:
                return False
        if i + rep < len(move):
            if move[i + rep][0] - move[i][0] != 1:
                return False

    return True


def _is_flush(move):
    return all(move[i][1] == move[i + 1][1] for i in range(0, len(move) - 1))
