if __name__ == "__main__":
    from vars import GLOBAL_LOG_FOLDER
    from ZhengShangYou.players.random_player import RandomPlayer
    from ZhengShangYou.zhengzero.zhengzero import ZhengZeroPlayer
    from ZhengShangYou.zhengzero.train import train_model

    player1 = ZhengZeroPlayer(
        player_id=0,
        model_path=f"{GLOBAL_LOG_FOLDER}/zheng-zero-0.pth",
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
    player2 = ZhengZeroPlayer(
        player_id=1,
        model_path=f"{GLOBAL_LOG_FOLDER}/zheng-zero-1.pth",
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
    player3 = RandomPlayer(player_id=2)
    player4 = RandomPlayer(player_id=3)

    players = [player1, player2, player3, player4]

    train_model(
        players,
        [0, 1],
        batch_size=100,
        params={
            "log": False,
        },
    )
