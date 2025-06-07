from ZhengShangYou.env.move_generator import MoveGenerator

import numpy as np


class BasePlayer:
    def __init__(self, player_id: int, player_name: str = None):
        self.player_id = player_id
        self.player_name = player_name if player_name else f"Player {player_id}"

    def play(self, info):
        """
        Play a card.
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
        """
        pass

    def replay(self, batch_size, replay_num, episode):
        """
        Replay the experiences in the replay buffer.
        """
        pass

    def _get_valid_moves(self, info):
        """
        Get the valid moves for the current player.
        """
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

        return valid_moves

    def _random_move(self, info):
        # Randomly select a card from the player's hand

        valid_moves = self._get_valid_moves(info)

        assert len(valid_moves) > 0, "Should have at least one valid move"

        cards_rem = len(info["cards"])
        for move in valid_moves:
            if len(move) == cards_rem:
                return move

        move_chances = [len(move) for move in valid_moves]

        move_chances = np.exp(move_chances) / np.sum(np.exp(move_chances))

        selected_move = valid_moves[np.random.choice(len(valid_moves), p=move_chances)]

        return selected_move
