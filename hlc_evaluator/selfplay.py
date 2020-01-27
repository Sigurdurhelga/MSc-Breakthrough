from monte_carlo.mcts import MCTS
from game_environments.play_chess.playchess import PlayChess
from game_environments.breakthrough.breakthrough import BTBoard

from collections import namedtuple

game_setup = namedtuple("gamesetup", "game_class initial_state")

BREAKTHROUGH_INITIAL_STATE = [
    [2, 2, 2],
    [0, 0, 0],
    [1, 1, 1]
]

CHESS_INITIAL_STATE = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

GAME = {
    "breakthrough": game_setup(BTBoard, BREAKTHROUGH_INITIAL_STATE),
    "chess": game_setup(PlayChess, CHESS_INITIAL_STATE),
}

def play_game(white_think, black_think):
    """
    Function that plays a game between two players
    using monte carlo tree search, white_think and
    black_think reference the amount of mcts rollout
    each player does
    """
    print("playgame started with {} {}".format(white_think, black_think))
    whiteplayer = MCTS()
    blackplayer = MCTS(negamaxing=True)

    selected_game = GAME["breakthrough"]

    board = selected_game.game_class(selected_game.initial_state, 1)
    while True:
        for _ in range(white_think):
            whiteplayer.do_rollout(board)
        board = whiteplayer.choose_child(board)
        # print("===================")
        # board.print_board()
        # print("===================")
        if board.is_terminal():
            break

        for _ in range(black_think):
            blackplayer.do_rollout(board)
        board = blackplayer.choose_child(board)
        print("===================")
        board.print_board()
        print("===================")
        if board.terminal:
            break
    return board.reward()

for j in range(24, 25):
    for k in range(1, 10):
        print(play_game(j, k))
