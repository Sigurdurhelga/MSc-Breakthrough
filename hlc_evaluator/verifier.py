from monte_carlo.mcts import MCTS
from monte_carlo.mctsnode import Node
# from game_environments.play_chess.playchess import PlayChess
from game_environments.gamenode import GameNode
from game_environments.breakthrough.breakthrough import BTBoard, config as BTconfig
from neural_networks.breakthrough.breakthrough_nn import BreakthroughNN

import numpy as np
from tqdm import tqdm

GAME = {
    "breakthrough": BTBoard(np.zeros([6,6]), 1),
}

selected_game = GAME["breakthrough"]
initial_state = selected_game.initial_state()

def selfplay(first_network_path, first_network_name, second_network_path, second_network_name, state_example):

  neural_network_1 = BreakthroughNN(state_example.cols, state_example.rows, state_example.get_move_amount())
  neural_network_2 = BreakthroughNN(state_example.cols, state_example.rows, state_example.get_move_amount())

  test_as_white = False

  if test_as_white:
    neural_network_1.loadmodel(first_network_path, first_network_name)
  else:
    neural_network_2.loadmodel(first_network_path, first_network_name)

  memo_nn1 = {}
  memo_nn2 = {}

  NN_THINK = 1

  initial_node = Node(state_example.initial_state(), "START")
  first_win = 0
  second_win = 0
  total_games = 0
  while True:

    for _ in tqdm(range(100)):
      curr_node = initial_node
      monte_tree_1 = MCTS()
      monte_tree_2 = MCTS()
      total_games += 1
      while True:

        # WHITE MOVES
        pi = monte_tree_1.get_policy(curr_node, NN_THINK, neural_network_1)
        
        if not curr_node.is_expanded():
          curr_node.expand()

        pi,v = neural_network_1.safe_predict(curr_node.gamestate)
        pi = pi.view(-1)

        for i,child in enumerate(curr_node.children):
          if not child:
            pi[i] = 0
        pi = pi / sum(pi).item()

        print("policy:",pi)
        print(sum(pi))
        curr_node = np.random.choice(curr_node.children, p=pi)

        if curr_node.gamestate.is_terminal():
          if curr_node.gamestate.reward() == 1:
            first_win += 1
          break

        # BLACK MOVES
        pi = monte_tree_2.get_policy(curr_node, NN_THINK, neural_network_2)
        if not curr_node.is_expanded():
          curr_node.expand()
        pi,v = neural_network_2.safe_predict(curr_node.gamestate)
        pi = pi.view(-1)

        for i,child in enumerate(curr_node.children):
          if not child:
            pi[i] = 0
        pi = pi / sum(pi).item()
        print("policy:",pi)
        print(sum(pi).item())

        curr_node = np.random.choice(curr_node.children, p=pi)

        if curr_node.gamestate.is_terminal():
          if curr_node.gamestate.reward() == -1:
            second_win += 1
          break
      if test_as_white:
        print("Bestnetwork {} random {} winrate {}".format(first_win, second_win, first_win / total_games))
      else:
        print("Bestnetwork {} random {} winrate {}".format(second_win, first_win, second_win / total_games))
    print("ENDGAME:")
    curr_node.gamestate.print_board()
    print("STATS AFTER EPISODE")
    if test_as_white:
      print("Bestnetwork {} random {} winrate {}".format(first_win, second_win, first_win / total_games))
    else:
      print("Bestnetwork {} random {} winrate {}".format(second_win, first_win, second_win / total_games))

network_path = "./trained_models"
selfplay(network_path,"test_network.tar", network_path, "random.tar",initial_state)
