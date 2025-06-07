from vars import GLOBAL_LOG_FOLDER
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
                        GLOBAL_LOG_FOLDER,
                        f"zhengzero.log",
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

logger = create_logger(__name__)


def _print_cards(cards, player=""):
    """
    Print the hand of cards.
    """
    if len(cards) == 0:
        logger.info(f"{player}: (pass)")
        return
    logger.info(
        f"{player}: {[f"{REVERSE_CARDS[card[0]]} of {REVERSE_SUITS[card[1]]}" for card in cards]}"
    )


def _print_card(card):
    """
    Print the card.
    """
    logger.info(f"{REVERSE_CARDS[card[0]]} of {REVERSE_SUITS[card[1]]}")


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
    """

    if trick is None:
        return 0
    else:
        return TRICKS[trick]


def int2trick(trick):
    """
    Convert the trick to an integer.
    """
    if trick == 0:
        return None
    else:
        return list(TRICKS.keys())[list(TRICKS.values()).index(trick)]


def string2card(card_str):
    """
    Convert a string representation of a card to a tuple (card, suit).
    """
    card_str = card_str.strip()
    card_str = card_str.split(" ")

    if card_str[0] == "LJ" or card_str[0] == "HJ":
        return (CARDS[card_str[0]], 0)

    return (CARDS[card_str[0]], SUITS[card_str[1]])


def input2cards(prompt):
    """
    Prompt the user for a list of cards and return them as a list of tuples.
    """
    while True:
        try:
            input_cards = input(prompt)

            if input_cards == "":
                return []

            input_cards = input_cards.split(",")

            input_cards = [string2card(card) for card in input_cards]

            # sort the cards to ensure consistent order
            input_cards.sort(key=lambda x: (x[0], x[1]))  # sort by rank then suit

            return input_cards

        except Exception as e:
            print("invalid input, please try again")
            continue
