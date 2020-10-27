from monte_carlo.mcts import MCTS
from monte_carlo.mctsnode import Node
# from game_environments.play_chess.playchess import PlayChess
from game_environments.breakthrough.breakthrough import BTBoard, config as BTconfig
from pprint import pprint
from collections import namedtuple
import numpy as np
from tqdm import tqdm
import json
from datetime import time


GAME = {
    "breakthrough": BTBoard(np.zeros([6,6]), 1),
}

selected_game = GAME["breakthrough"]
initial_state = selected_game.initial_state()

testerino = Node(selected_game.initial_state(),"START")


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
    curr_node = Node(selected_game.initial_state(), "START")

    whiteplayer = MCTS()
    blackplayer = MCTS()


    while True:
        for _ in range(white_think):
            whiteplayer.rollout(curr_node)

        curr_node = whiteplayer.get_best_child(curr_node)

        if curr_node.gamestate.is_terminal():
            break

        # print("blackplayer Q for currnode before rollout:",blackplayer.Qs[curr_node])
        for _ in range(black_think):
            blackplayer.rollout(curr_node)
            # print("blackplayer Q for currnode after rollout:",blackplayer.Qs[curr_node])
        curr_node = blackplayer.get_best_child(curr_node)

        if curr_node.gamestate.is_terminal():
            break

    print("endgame")
    curr_node.gamestate.print_board()

    return curr_node.gamestate.reward()

results_start = {
    "black": 0,
    "white": 0,
}

whitewins = 0
blackwins = 0

for _ in tqdm(range(100)):
    winner = play_game(30,50)
    if winner == 1:
        whitewins += 1
    else:
        blackwins += 1
    print("White {} - black {}".format(whitewins, blackwins))
