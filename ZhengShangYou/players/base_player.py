from ZhengShangYou.env.move_generator import MoveGenerator, _is_bomb

import numpy as np

from collections import deque


class BasePlayer:
    def __init__(self, player_id: int):
        self.player_id = player_id

    def play(self, info):
        """
        Play a card.
        :param info: The game information
        :return: The card to be played
        """

        selected_move = self._play(info)
        return selected_move

    def _play(self, info):
        """
        Play a card.
        :param info: The game information
        :return: The card to be played
        """
        raise NotImplementedError("This method should be overridden by subclasses")

    def remember(self, state, action, reward, done):
        """
        Store the experience in a replay buffer.
        :param state: The current state
        :param action: The action taken
        :param reward: The reward received
        :param next_state: The next state
        :param done: Whether the episode is done
        """
        pass

    def replay(self, batch_size, replay_num, episode):
        """
        Replay the experiences in the replay buffer.
        """
        pass

    def _random_move(self, info):
        # Randomly select a card from the player's hand

        trick = info["trick"]
        last_played_cards = info["last_played_cards"]

        move_gen = MoveGenerator(info["cards"])

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

        if len(valid_moves) >= 2:
            move_chances = [1.0 for _ in valid_moves]

            length = 0
            num = 0
            for i in range(len(move_chances)):
                if len(valid_moves[i]) != length:
                    move_chances[i - 1] = length
                    length = len(valid_moves[i])
                    num = 0

                move_chances[i] = length * np.power(0.9, num)

            move_chances = np.exp(move_chances) / np.sum(np.exp(move_chances))

            selected_move = valid_moves[
                np.random.choice(len(valid_moves), p=move_chances)
            ]

        return selected_move
