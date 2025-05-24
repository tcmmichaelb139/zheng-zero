import logging
from rich.logging import RichHandler
from rich import print
import os
import time


def create_logger(name):
    logging.basicConfig(
        level="INFO",
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            RichHandler(rich_tracebacks=True),
            (
                logging.FileHandler(
                    os.path.join(
                        "ZhengShangYou/zhengzero/logs",
                        f"zhengzero_{int(time.time())}.log",
                    ),
                    mode="a",
                )
            ),
        ],
    )

    return logging.getLogger(name)


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
REVERSE_CARDS = {v: k for k, v in CARDS.items()}
REVERSE_SUITS = {v: k for k, v in SUITS.items()}


def _print_cards(cards, player=""):
    """
    Print the hand of cards.
    :param cards: The hand of cards
    """
    if len(cards) == 0:
        print(f"{player}: (pass)")
        return
    print(
        f"{player}: {[f"{REVERSE_CARDS[card[0]]} of {REVERSE_SUITS[card[1]]}" for card in cards]}"
    )


def _print_card(card):
    """
    Print the card.
    :param card: The card to be printed
    """
    print(f"{REVERSE_CARDS[card[0]]} of {REVERSE_SUITS[card[1]]}")


def card2int(card):
    """
    Convert the card to an integer.
    """
    if card[0] == 13 or card[0] == 14:
        return 52 + card[0] - 13
    else:
        return card[0] * 4 + card[1]


def int2card(card):
    """
    Convert the integer to a card.
    """
    if card >= 52:
        return (card - 52 + 13, 0)
    else:
        return (card // 4, card % 4)


TRICKS = {
    "single": 1,
    "pair": 2,
    "triple": 3,
    "straight_1": 4,
    "straight_22": 5,
    "straight_333": 6,
    "bomb": 7,
}


def trick2int(trick):
    """
    Convert the trick to an integer.
    :param trick: The trick to be converted
    :return: The trick as an integer
    """
    if trick == "pass" or trick is None:
        return 0
    else:
        return TRICKS[trick]


def int2trick(trick):
    """
    Convert the trick to an integer.
    :param trick: The trick to be converted
    :return: The trick as an integer
    """
    if trick == 0:
        return "pass"
    else:
        return list(TRICKS.keys())[list(TRICKS.values()).index(trick)]
