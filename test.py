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
        model_path="ZhengShangYou/zhengzero/models/zheng-zero-1.pth",
        model_params={
            "lr": 0.00001,
            "gamma": 1,
            "epsilon": 0.00,
            "epsilon_min": 0.00,
            "epsilon_decay": 0.99,
            "update_length": 1000,
            "replay_num": 3,
            "train": False,
            "save_model": True,
        },
        buffer_size=1000000,
    )

    player2 = ZhengZeroPlayer(
        player_id=1,
        model_path="ZhengShangYou/zhengzero/models/zheng-zero-1.pth",
        model_params={
            "lr": 0.00001,
            "gamma": 0.99,
            "epsilon": 0,
            "epsilon_min": 0.00,
            "epsilon_decay": 0.99,
            "update_length": 1000,
            "replay_num": 3,
            "train": False,
            "save_model": True,
        },
        buffer_size=1000000,
    )
    player3 = RandomPlayer(player_id=2)
    player4 = RandomPlayer(player_id=3)

    players = [player1, player2, player3, player4]

    train_model(
        players,
        [1],
        batch_size=64,
        params={
            "train": True,
            "log": True,
        },
    )
