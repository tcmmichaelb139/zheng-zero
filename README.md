# Zheng Shang You

## Rules

Different from [wikipedia Zheng Shang You](https://en.wikipedia.org/wiki/Zheng_Shangyou)

Patterns:

- singles
- doubles
- triples
- straight (3 or more cards in a row) (same suit is higher than different suit)
- double straight (3 or more pairs in a row) (same suit is higher than different suit)
- triple straight (3 or more triples in a row) (same suit is higher than different suit)
- bomb (4 cards of the same rank)

## Algorithms

### Basic

DQN with 54 observations (one for every card) and certain actions:

- play (pattern) lowest card
- play (pattern) highest card
- pass

### Advanced

DQN with 54 observations (one for every card) and 55 actions (one for every card + one for pass).

We select the action based on the
