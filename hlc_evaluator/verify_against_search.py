from monte_carlo.mcts import MCTS
from monte_carlo.mctsnode import Node
# from game_environments.play_chess.playchess import PlayChess
from game_environments.gamenode import GameNode
from game_environments.breakthrough.breakthrough import BTBoard, config as BTconfig
from neural_networks.breakthrough.breakthrough_nn import BreakthroughNN
from search_algorithms.alphabeta import alpha_beta_search

import torch

import numpy as np
from tqdm import tqdm

GAME = {
    "breakthrough": BTBoard(np.zeros([6,6]), 1),
}

selected_game = GAME["breakthrough"]
initial_state = selected_game.initial_state()

CUDA = torch.cuda.is_available()

def selfplay(first_network_path, first_network_name, state_example):

  neural_network_1 = BreakthroughNN(state_example.cols, state_example.rows, state_example.get_move_amount())

  test_as_white = True

  neural_network_1.loadmodel(first_network_path, first_network_name)
  SEARCH_DEPTH = 3
  initial_node = Node(state_example.initial_state(), "START")
  first_win = 0
  second_win = 0
  total_games = 0
  while True:
    for _ in range(20):
      curr_node = initial_node
      monte_tree_1 = MCTS()
      total_games += 1
      while True:
        if test_as_white:
          if not curr_node.is_expanded():
            curr_node.expand()
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
          curr_node = np.random.choice(curr_node.children, p=pi)

          if curr_node.gamestate.is_terminal():
            if curr_node.gamestate.reward() == 1:
              first_win += 1
            break
        else:
          next_node,move = alpha_beta_search(curr_node.gamestate, SEARCH_DEPTH, "player_piece_amount")
          curr_node = Node(next_node, move)
          if curr_node.gamestate.is_terminal():
            if curr_node.gamestate.reward() == 1:
              first_win += 1
            break

        if not test_as_white:
          if not curr_node.is_expanded():
            curr_node.expand()

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
          curr_node = np.random.choice(curr_node.children, p=pi)

          if curr_node.gamestate.is_terminal():
            if curr_node.gamestate.reward() == -1:
              second_win += 1
            break
        else:
          next_node,move = alpha_beta_search(curr_node.gamestate, SEARCH_DEPTH, "player_piece_amount")
          curr_node = Node(next_node, move)
          if curr_node.gamestate.is_terminal():
            if curr_node.gamestate.reward() == 1:
              second_win += 1
            break
    winrate = first_win / total_games if test_as_white else second_win / total_games
    return winrate

network_path = "./trained_models"
for i in range(23,31):
  print(i*10,selfplay(network_path,f"session_res3_gen{i*10}.tar",initial_state))