from .move_generator import _is_straight, _is_bomb


def detect_move(move):
    """
    Detect the move type.
    """
    if len(move) == 0:
        return None
    elif len(move) == 1:
        return "single"
    elif len(move) == 2:
        if move[0][0] == move[1][0]:
            return "pair"
        else:
            assert False, "Invalid move: two different cards"
    elif len(move) == 3:
        if move[0][0] == move[1][0] == move[2][0]:
            return "triple"
        elif _is_straight(move):
            return "straight_1"
        else:
            assert False, "Invalid move: three different cards"
    elif len(move) == 4:
        if _is_bomb(move):
            return "bomb"
        elif _is_straight(move):
            return "straight_1"
        else:
            assert False, "Invalid move: four different cards"
    elif len(move) >= 5:
        if _is_straight(move):
            return "straight_1"
        elif _is_straight(move, 2):
            return "straight_22"
        elif _is_straight(move, 3):
            return "straight_333"
        else:
            assert False, "Invalid move: five or more different cards"
    else:
        return "invalid"
