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

- model is not winning completely yet, going to try some other models

current model uses 224 inputs and one output (q value):

- 8 for the trick
- 54 for the last played hand
- 54 for all the played cards
- 54 for the current hand
- 54 for the action

current model gets 1st/2nd ~70-80% and loses < 10% of the time against greedy random players (weighted by length of the valid hands played)

tested to around 20000 episodes out of 100000
