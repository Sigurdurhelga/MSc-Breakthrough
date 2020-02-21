from monte_carlo.mcts import MCTS
from monte_carlo.mctsnode import Node
from game_environments.play_chess.playchess import PlayChess
from game_environments.breakthrough.breakthrough import BTBoard, config as BTconfig

from collections import namedtuple
import numpy as np
from tqdm import tqdm
import json
from datetime import time

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

BREAKTHROUGH_INITIAL_STATE = (np.array([
                  [BTconfig.BLACK,BTconfig.BLACK,BTconfig.BLACK],
                  [BTconfig.BLACK,BTconfig.BLACK,BTconfig.BLACK],
                  [BTconfig.EMPTY,BTconfig.EMPTY,BTconfig.EMPTY],
                  [BTconfig.EMPTY,BTconfig.EMPTY,BTconfig.EMPTY],
                  [BTconfig.EMPTY,BTconfig.EMPTY,BTconfig.EMPTY],
                  [BTconfig.WHITE,BTconfig.WHITE,BTconfig.WHITE],
                  [BTconfig.WHITE,BTconfig.WHITE,BTconfig.WHITE]]), BTconfig.WHITE)


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
    # print("playgame started with {} {}".format(white_think, black_think))
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
        # selection / expantion / rollout
        whiteplayer.rollout(white_think)
        # greedy select best
        whiteplayer.move_to_best_child()
        curr_node = whiteplayer.root

        if verbose:
            print("did action: ",whiteplayer.root.action)
            whiteplayer.root.gamestate.print_board()
            print("===================")
        if whiteplayer.root.gamestate.is_terminal():
            break

        if not blackplayer:
            blackplayer = MCTS(Node(whiteplayer.root.gamestate, whiteplayer.root.action), negamaxing=True)
        else:
            blackplayer.move_to_child(whiteplayer.root.action)

        if verbose:
            print("===================")
            blackplayer.root.gamestate.print_board()

        # selection / expantion / rollout
        blackplayer.rollout(black_think)
        # greedy select best
        blackplayer.move_to_best_child()
        curr_node = blackplayer.root

        if verbose:
            print("did action: ",blackplayer.root.action)
            blackplayer.root.gamestate.print_board()
            print("===================")
        if blackplayer.root.gamestate.is_terminal():
            break

        whiteplayer.move_to_child(blackplayer.root.action)
    return curr_node.gamestate.reward()

results_start = {
    "black": 0,
    "white": 0,
    "tie": 0,
}
total_results = {}

for i in tqdm(range(1,10)):
    results = results_start.copy()
    for _ in tqdm(range(10)):
        winner = play_game(20,5)
        if winner == 0:
            results["tie"] += 1
        elif winner == 1:
            results["white"] += 1
        else:
            results["black"] += 1
    total_results[f"1.{i}"] = results

for i in tqdm(range(1,10)):
    results = results_start.copy()
    for _ in tqdm(range(10)):
        winner = play_game(50,5)
        if winner == 0:
            results["tie"] += 1
        elif winner == 1:
            results["white"] += 1
        else:
            results["black"] += 1
    total_results[f"2.{i}"] = results
with open("results5.json", "w") as f:
    f.write(json.dumps(total_results))
