from monte_carlo.mcts import MCTS
from monte_carlo.mctsnode import Node
# from game_environments.play_chess.playchess import PlayChess
from game_environments.gamenode import GameNode
from game_environments.breakthrough.breakthrough import BTBoard, config as BTconfig
from neural_networks.breakthrough.breakthrough_nn import BreakthroughNN

import torch

import numpy as np
from tqdm import tqdm

GAME = {
    "breakthrough": BTBoard(np.zeros([6,6]), 1),
}

selected_game = GAME["breakthrough"]
initial_state = selected_game.initial_state()

CUDA = torch.cuda.is_available()

def selfplay(first_network_path, first_network_name, second_network_path, second_network_name, state_example):

  neural_network_1 = BreakthroughNN(state_example.cols, state_example.rows, state_example.get_move_amount())
  neural_network_2 = BreakthroughNN(state_example.cols, state_example.rows, state_example.get_move_amount())

  test_as_white = True

  if test_as_white:
    neural_network_1.loadmodel(first_network_path, first_network_name)
    neural_network_2.loadmodel(second_network_path, second_network_name)
  else:
    neural_network_2.loadmodel(first_network_path, first_network_name)
    neural_network_1.loadmodel(second_network_path, second_network_name)

  memo_nn1 = {}
  memo_nn2 = {}

  initial_node = Node(state_example.initial_state(), "START")
  first_win = 0
  second_win = 0
  total_games = 0
  while total_games < 200:

    for _ in tqdm(range(5)):
      curr_node = initial_node
      monte_tree_1 = MCTS()
      monte_tree_2 = MCTS()
      total_games += 1
      while True:

        # WHITE MOVES
        # pi = monte_tree_1.get_policy(curr_node, NN_THINK, neural_network_1,0)

        
        if not curr_node.is_expanded():
          curr_node.expand()
        # for i,child in enumerate(curr_node.children):
          # if not child:
            # pi[i] = 0
        # pi = pi / sum(pi).item()

        # """
        pi,v = neural_network_1.safe_predict(curr_node.gamestate)
        if CUDA:
          pi = pi.detach().cpu().numpy() 
          pi = pi.reshape(-1)
        else:
          pi = pi.view(-1)

        for i,child in enumerate(curr_node.children):
          if not child:
            pi[i] = 0
        pi = pi / sum(pi).item()

        # print("policy:",pi)
        # print(sum(pi))
        # """
        curr_node = np.random.choice(curr_node.children, p=pi)

        if curr_node.gamestate.is_terminal():
          if curr_node.gamestate.reward() == 1:
            first_win += 1
          break

        # BLACK MOVES
        # pi = monte_tree_2.get_policy(curr_node, NN_THINK, neural_network_2,0)
        
        if not curr_node.is_expanded():
          curr_node.expand()

        # for i,child in enumerate(curr_node.children):
          # if not child:
            # pi[i] = 0
        # pi = pi / sum(pi).item()
        # """
        pi,v = neural_network_2.safe_predict(curr_node.gamestate)
        if CUDA:
          pi = pi.detach().cpu().numpy() 
          pi = pi.reshape(-1)
        else:
          pi = pi.view(-1)

        for i,child in enumerate(curr_node.children):
          if not child:
            pi[i] = 0
        pi = pi / sum(pi).item()
        # print("policy:",pi)
        # print(sum(pi).item())

        # """

        curr_node = np.random.choice(curr_node.children, p=pi)

        if curr_node.gamestate.is_terminal():
          if curr_node.gamestate.reward() == -1:
            second_win += 1
          break
      # if test_as_white:
        # print("Bestnetwork {} random {} winrate {}".format(first_win, second_win, first_win / total_games))
      # else:
        # print("Bestnetwork {} random {} winrate {}".format(second_win, first_win, second_win / total_games))

    winrate = first_win / total_games if test_as_white else second_win / total_games
    return winrate

network_path = "./trained_models"
for i in range(1,31):
    print(selfplay(network_path,f"session_res3_gen{i*10}.tar", network_path, "random1234.tar",initial_state))
