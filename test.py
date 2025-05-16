if __name__ == "__main__":
    from ZhengShangYou.env.move_generator import MoveGenerator, _is_straight
    from ZhengShangYou.env.zhengshangyou import ZhengShangYou
    from ZhengShangYou.players.random_player import RandomPlayer
    from ZhengShangYou.zhengzero.zhengzero import ZhengZeroPlayer, train_model
    from ZhengShangYou.env.utils import _print_cards
    import torch

    # gen = MoveGenerator([(0, 0), (1, 0), (2, 0), (3, 1)])

    # print(
    #     gen._valid_move(
    #         "straight_1",
    #         [(2, 0), (3, 0), (4, 0)],
    #         [(3, 1), (4, 1), (5, 1)],
    #     )
    # )

    player1 = ZhengZeroPlayer(
        player_id=0,
        model_params={
            "lr": 0.01,
            "gamma": 0.99,
            "epsilon": 1,
            "epsilon_min": 0.01,
            "epsilon_decay": 0.99,
        },
        buffer_size=10000,
    )

    player2 = ZhengZeroPlayer(
        player_id=1,
        model_params={
            "lr": 0.01,
            "gamma": 0.99,
            "epsilon": 1,
            "epsilon_min": 0.01,
            "epsilon_decay": 0.99,
        },
        buffer_size=10000,
    )
    player3 = RandomPlayer(player_id=2)
    player4 = RandomPlayer(player_id=3)

    players = [player1, player2, player3, player4]

    train_model(
        players,
        [0],
        batch_size=64,
        num_episodes=10000,
    )
