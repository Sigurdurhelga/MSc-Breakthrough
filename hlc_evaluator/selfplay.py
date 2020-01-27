from monte_carlo.mcts import MCTS
from game_environments.play_chess.playchess import PlayChess
from game_environments.breakthrough.breakthrough import BTBoard

# Change GAME variable to change selfplayed game
GAME = BTBoard

def play_game(white_think, black_think):
    print("playgame started with {} {}".format(white_think, black_think))
    whiteplayer = MCTS()
    blackplayer = MCTS(negamaxing=True)
    board = GAME()
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
    return board.board.result(claim_draw=True)

for j in range(24,25):
    for k in range(1,10):
        print(play_game(j,k))





