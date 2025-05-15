if __name__ == "__main__":
    from ZhengShangYou.env.move_generator import MoveGenerator, _is_straight
    from ZhengShangYou.env.env import Env
    from ZhengShangYou.players.random_player import RandomPlayer
    from ZhengShangYou.env.utils import _print_cards

    # gen = MoveGenerator([(0, 0), (1, 0), (2, 0), (3, 1)])

    # print(
    #     gen._valid_move(
    #         "straight_1",
    #         [(2, 0), (3, 0), (4, 0)],
    #         [(3, 1), (4, 1), (5, 1)],
    #     )
    # )

    player1 = RandomPlayer("player1")
    player2 = RandomPlayer("player2")
    player3 = RandomPlayer("player3")
    player4 = RandomPlayer("player4")

    env = Env([player1, player2, player3, player4])

    env._deal_cards()

    for player in env.players:
        print(len(player.cards))
        _print_cards(player.cards)
        print()

    env.step()
