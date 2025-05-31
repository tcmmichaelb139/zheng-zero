from ZhengShangYou.zhengzero.simulate import sim_model

if __name__ == "__main__":
    from ZhengShangYou.players.random_player import RandomPlayer
    from ZhengShangYou.players.input_player import InputPlayer
    from ZhengShangYou.zhengzero.zhengzero import ZhengZeroPlayer

    players = [
        ZhengZeroPlayer(
            player_id=0,
            model_path="ZhengShangYou/zhengzero/logs/2025-05-30_23-22-10/models/3/zheng-zero-94000.pth",
            model_params={
                "epsilon": 0.0,
                "epsilon_min": 0.0,
                "epsilon_decay": 0.999,
                "train": False,
                "save_model": False,
            },
        ),
        ZhengZeroPlayer(
            player_id=1,
            model_path="ZhengShangYou/zhengzero/logs/2025-05-30_16-27-01/models/0/zheng-zero-106000.pth",
            model_params={
                "epsilon": 0.0,
                "epsilon_min": 0.0,
                "epsilon_decay": 0.999,
                "train": False,
                "save_model": False,
            },
        ),
        ZhengZeroPlayer(
            player_id=2,
            model_path="ZhengShangYou/zhengzero/logs/2025-05-30_23-22-10/models/3/zheng-zero-94000.pth",
            model_params={
                "epsilon": 0.0,
                "epsilon_min": 0.0,
                "epsilon_decay": 0.999,
                "train": False,
                "save_model": False,
            },
        ),
        ZhengZeroPlayer(
            player_id=3,
            model_path="ZhengShangYou/zhengzero/logs/2025-05-30_16-27-01/models/0/zheng-zero-106000.pth",
            model_params={
                "epsilon": 0.0,
                "epsilon_min": 0.0,
                "epsilon_decay": 0.999,
                "train": False,
                "save_model": False,
            },
        ),
    ]

    sim_model(players, num_episodes=1000, log=False)
