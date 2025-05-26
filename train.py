if __name__ == "__main__":
    from ZhengShangYou.players.random_player import RandomPlayer
    from ZhengShangYou.zhengzero.zhengzero import ZhengZeroPlayer, train_model

    player1 = RandomPlayer(player_id=0)
    player2 = RandomPlayer(player_id=1)
    player3 = ZhengZeroPlayer(
        player_id=2,
        model_path="ZhengShangYou/zhengzero/models/zheng-zero-2.pth",
        model_params={
            "lr": 0.00001,
            "gamma": 0.99,
            "epsilon": 1,
            "epsilon_min": 0.1,
            "epsilon_decay": 0.999,
            "update_length": 1000,
            "replay_num": 3,
            "train": True,
            "save_model": True,
        },
        buffer_size=1000000,
    )
    player4 = RandomPlayer(player_id=3)

    players = [player1, player2, player3, player4]

    train_model(
        players,
        [2],
        batch_size=64,
        params={
            "train": True,
            "log": False,
            "sim": False,
        },
    )
