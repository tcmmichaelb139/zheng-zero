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


def _print_cards(cards):
    """
    Print the hand of cards.
    :param cards: The hand of cards
    """
    if len(cards) == 0:
        print("No cards (pass)")
        return
    for card in cards:
        print(f"{REVERSE_CARDS[card[0]]} of {REVERSE_SUITS[card[1]]}", end=", ")
    print()


def _print_card(card):
    """
    Print the card.
    :param card: The card to be printed
    """
    print(f"{REVERSE_CARDS[card[0]]} of {REVERSE_SUITS[card[1]]}")
