if __name__ == "__main__":
    from vars import GLOBAL_LOG_FOLDER
    from ZhengShangYou.zhengzero.zhengzero import ZhengZeroPlayer
    from ZhengShangYou.zhengzero.train import self_play

    player = ZhengZeroPlayer(
        player_id=0,
        model_params={
            "lr": 0.00001,
            "gamma": 0.99,
            "epsilon": 1,
            "epsilon_min": 0.1,
            "epsilon_decay": 0.999,
            "update_length": 10000,
            "replay_num": 1,
            "train": True,
            "save_model": True,
        },
        buffer_size=1000000,
    )

    self_play(player, pool_size=25, batch_size=100, update_length=1000)
