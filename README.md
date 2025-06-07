# Zheng Shang You

## Rules

This is a different from [wikipedia Zheng Shang You](https://en.wikipedia.org/wiki/Zheng_Shangyou)

Tricks:

- singles
- doubles
- triples
- straight: 3 or more cards in a row
- double straight: 3 or more pairs in a row
- triple straight: 3 or more triples in a row
- bomb: 4 cards of the same rank

Straights that are all the same suit are of higher rank than straights that are not. (If multiple decks are used double/triple straights are possible)

## Algorithms

Double Deep Q-Learning (DDQN) were used to train the models. The model has 224 inputs and one output (Q value):

- 8 for the trick
- 54 for the last played hand
- 54 for all the played cards
- 54 for the current hand
- 54 for the action

A possible addition is using an LSTM model for historical cards.

For the replay buffer, I used Prioritized Experience Replay (PER) to improve the efficiency of learning by sampling more important experiences more frequently (or weighting more important experiences higher is how I understood it)

Rewards are as follows:

- 0.25 for playing a card
- 1.00 for winning the entire game
- 0 for placing 2nd - 4th

Setting rewards to be between 0 and 1 helps the model from having exploading gradients (which happened in the past implementations/commits).

There were two strategies of training that I implemented:

### Bot training

The model plays against random (greedy) strategy players or other trained models. This way the model learns to play optimally against the certain strategy.

### Self-play

The model plays against itself, or more specifically, against a pool of past versions of itself.

The strategy for choosing which versions to save in the pool was based on the [TrueSkill](https://en.wikipedia.org/wiki/TrueSkill) rating system. If the current model's rating is higher than the lowest in the pool, it replaces that model. This ensures that the pool contains only the best-performing models.

Every episode 3 other models are selected to play against the current model.

# References

- https://arxiv.org/abs/2106.06135
