from ZhengShangYou.env.zhengshangyou import ZhengShangYou
from ZhengShangYou.env.utils import _print_cards


def sim_model(
    players,
    zheng_zero_players,
):

    env = ZhengShangYou(
        players,
        {
            "train": False,
            "log": False,
            "sim": True,
            "zhengzeroplayers": zheng_zero_players,
        },
    )

    state = env.reset()
    done = False

    while not done:
        current_player = env._current_player()
        action = players[current_player].play(state)

        _print_cards(action, current_player)

        next_state, _, done = env.step(action)

        state = next_state


if __name__ == "__main__":
    from ZhengShangYou.players.input_player import InputPlayer
    from ZhengShangYou.zhengzero.zhengzero import ZhengZeroPlayer, train_model

    player1 = ZhengZeroPlayer(
        player_id=0,
        model_path="ZhengShangYou/zhengzero/models/zheng-zero-0.pth",
        model_params={
            "lr": 0.00001,
            "gamma": 0.97,
            "epsilon": 1,
            "epsilon_min": 0.1,
            "epsilon_decay": 0.99,
            "train": False,
            "save_model": True,
        },
        buffer_size=1000000,
    )

    player2 = InputPlayer(player_id=1)
    player3 = InputPlayer(player_id=2)
    player4 = InputPlayer(player_id=3)

    players = [player1, player2, player3, player4]
    zheng_zero_players = [0]

    sim_model(players, zheng_zero_players)
