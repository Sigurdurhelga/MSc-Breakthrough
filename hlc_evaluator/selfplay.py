from monte_carlo.mcts import MCTS
from monte_carlo.mctsnode import Node
from game_environments.play_chess.playchess import PlayChess
from game_environments.breakthrough.breakthrough import BTBoard, config as BTconfig

from collections import namedtuple
import numpy as np

game_setup = namedtuple("gamesetup", "game_class initial_state")

BREAKTHROUGH_INITIAL_STATE = (np.array([
                  [BTconfig.BLACK,BTconfig.BLACK,BTconfig.BLACK,BTconfig.BLACK],
                  [BTconfig.BLACK,BTconfig.BLACK,BTconfig.BLACK,BTconfig.BLACK],
                  [BTconfig.EMPTY,BTconfig.EMPTY,BTconfig.EMPTY,BTconfig.EMPTY],
                  [BTconfig.EMPTY,BTconfig.EMPTY,BTconfig.EMPTY,BTconfig.EMPTY],
                  [BTconfig.EMPTY,BTconfig.EMPTY,BTconfig.EMPTY,BTconfig.EMPTY],
                  [BTconfig.EMPTY,BTconfig.EMPTY,BTconfig.EMPTY,BTconfig.EMPTY],
                  [BTconfig.WHITE,BTconfig.WHITE,BTconfig.WHITE,BTconfig.WHITE],
                  [BTconfig.WHITE,BTconfig.WHITE,BTconfig.WHITE,BTconfig.WHITE]]), BTconfig.WHITE)

CHESS_INITIAL_STATE = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

GAME = {
    "breakthrough": game_setup(BTBoard, BREAKTHROUGH_INITIAL_STATE),
    "chess": game_setup(PlayChess, CHESS_INITIAL_STATE),
}

selected_game = GAME["breakthrough"]
initial_state = Node(selected_game.game_class(*selected_game.initial_state), "START")


whiteplayer = None
blackplayer = None

def play_game(white_think, black_think, verbose=False):
    global whiteplayer, blackplayer, selected_game
    """
    Function that plays a game between two players
    using monte carlo tree search, white_think and
    black_think reference the amount of mcts rollout
    each player does
    """
    print("playgame started with {} {}".format(white_think, black_think))
    # initial node (root)
    curr_node = Node(selected_game.game_class(*selected_game.initial_state), "START")
    if whiteplayer:
        whiteplayer.set_node(curr_node)
        blackplayer.set_node(curr_node)
    else:
        whiteplayer = MCTS(curr_node)

    while True:
        if verbose:
            print("===================")
            whiteplayer.root.gamestate.print_board()
        whiteplayer.rollout(white_think)
        whiteplayer.move_to_best_child()
        if verbose:
            print("did action: ",whiteplayer.root.action)
            whiteplayer.root.gamestate.print_board()
            print("===================")
        if whiteplayer.root.gamestate.is_terminal():
            print("terminal after whitemove")
            break

        if not blackplayer:
            blackplayer = MCTS(Node(whiteplayer.root.gamestate, whiteplayer.root.action), negamaxing=True)
        else:
            blackplayer.move_to_child(whiteplayer.root.action)

        if verbose:
            print("===================")
            blackplayer.root.gamestate.print_board()
        blackplayer.rollout(black_think)
        blackplayer.move_to_best_child()
        if verbose:
            print("did action: ",blackplayer.root.action)
            blackplayer.root.gamestate.print_board()
            print("===================")
        if blackplayer.root.gamestate.is_terminal():
            print("terminal after blackmove")
            break
        whiteplayer.move_to_child(blackplayer.root.action)

    #print("END RESULTS WHITE")
    #print(whiteplayer.Ns)
    #print(whiteplayer.Qs)
    #print("END RESULTS BLACK")
    #print(blackplayer.Ns)
    #print(blackplayer.Qs)
    return curr_node.gamestate.reward()
wins = {
    -1: 0,
    0: 0,
    1: 0
}
for i in range(3):
    wins[play_game(80, 1)] += 1
    print(wins)
    print("INFO")
    print("Whiteplayer initial state info", whiteplayer.Qs[initial_state], whiteplayer.Ns[initial_state])
    print("blackplayer initial state info", blackplayer.Qs[initial_state], blackplayer.Ns[initial_state])
    print("END RESULTS BLACK")
print(wins)
