from ZhengShangYou.zhengzero.simulate import sim_model

if __name__ == "__main__":
    from ZhengShangYou.players.random_player import RandomPlayer
    from ZhengShangYou.players.input_player import InputPlayer
    from ZhengShangYou.zhengzero.zhengzero import ZhengZeroPlayer

    players = [
        ZhengZeroPlayer(
            player_id=0,
            model_path="ZhengShangYou/zhengzero/logs/2025-06-06_20-57-47/models/0/zheng-zero-6520000.pth",
            model_params={
                "epsilon": 1.0,
                "epsilon_min": 1.0,
                "epsilon_decay": 0.999,
                "train": False,
                "save_model": False,
            },
        ),
        ZhengZeroPlayer(
            player_id=1,
            model_path="ZhengShangYou/zhengzero/logs/2025-06-06_20-57-47/models/0/zheng-zero-10880000.pth",
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
            model_path="ZhengShangYou/zhengzero/logs/2025-06-06_20-57-47/models/0/zheng-zero-6520000.pth",
            model_params={
                "epsilon": 1.0,
                "epsilon_min": 1.0,
                "epsilon_decay": 0.999,
                "train": False,
                "save_model": False,
            },
        ),
        ZhengZeroPlayer(
            player_id=3,
            model_path="ZhengShangYou/zhengzero/logs/2025-06-06_20-57-47/models/0/zheng-zero-10880000.pth",
            model_params={
                "epsilon": 0.0,
                "epsilon_min": 0.0,
                "epsilon_decay": 0.999,
                "train": False,
                "save_model": False,
            },
        ),
        # InputPlayer(player_id=3, player_name="Input Player"),  # Human player
    ]

    sim_model(players, num_episodes=10000, log=False)
