from monte_carlo.mcts import MCTS
from monte_carlo.mctsnode import Node
from game_environments.play_chess.playchess import PlayChess
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

  test_as_white = True

  if test_as_white:
    neural_network_1.loadmodel(first_network_path, first_network_name)
  else:
    neural_network_2.loadmodel(first_network_path, first_network_name)

  initial_node = Node(state_example.initial_state(), "START")
  first_win = 0
  second_win = 0
  total_games = 0
  while True:


    for _ in tqdm(range(100)):
      curr_node = initial_node
      while True:
        # First NN moves
        pi,_ = neural_network_1.predict(curr_node.gamestate)
        pi = pi.detach().cpu().numpy().reshape(-1)
        if not curr_node.is_expanded():
          curr_node.expand()

        action_idxs = [child.get_pidx() for child in curr_node.children if child]
        mask_idxs = [i for i in range(len(pi)) if i not in action_idxs]
        pi[mask_idxs] = 0
        # curr_node.gamestate.print_board()

        total = sum(pi)
        if total == 0:
          pi[action_idxs[0]] = 1.0
        else:
          pi = pi / total

        curr_node = np.random.choice(curr_node.children, p=pi)

        if curr_node.gamestate.is_terminal():
          if curr_node.gamestate.reward() == 1:
            first_win += 1
          break

        # Second NN moves (is playing black)
        pi,_ = neural_network_2.predict(curr_node.gamestate)
        pi = pi.detach().cpu().numpy().reshape(-1)
        if not curr_node.is_expanded():
          curr_node.expand()

        action_idxs = [child.get_pidx() for child in curr_node.children if child]
        mask_idxs = [i for i in range(len(pi)) if i not in action_idxs]
        pi[mask_idxs] = 0

        # curr_node.gamestate.print_board()

        total = sum(pi)
        if total == 0:
          pi[action_idxs[0]] = 1.0
        else:
          pi = pi / total

        curr_node = np.random.choice(curr_node.children, p=pi)
        if curr_node.gamestate.is_terminal():
          if curr_node.gamestate.reward() == -1:
            second_win += 1
          break
    print("ENDGAME:")
    curr_node.gamestate.print_board()
    print("STATS AFTER EPISODE")
    total_games = first_win + second_win
    if test_as_white:
      print("Bestnetwork {} random {} winrate {}".format(first_win, second_win, first_win / total_games))
    else:
      print("Bestnetwork {} random {} winrate {}".format(second_win, first_win, second_win / total_games))

network_path = "./trained_models"
selfplay(network_path,"totest.tar", network_path, "random.tar",initial_state)
