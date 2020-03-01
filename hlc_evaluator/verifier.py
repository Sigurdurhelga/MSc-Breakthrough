from monte_carlo.mcts import MCTS
from monte_carlo.mctsnode import Node
# from game_environments.play_chess.playchess import PlayChess
from game_environments.gamenode import GameNode
from game_environments.breakthrough.breakthrough import BTBoard, config as BTconfig
from neural_networks.breakthrough.breakthrough_nn import BreakthroughNN

import numpy as np
from tqdm import tqdm

GAME = {
    "breakthrough": BTBoard(np.zeros([5,4]), 1),
}

selected_game = GAME["breakthrough"]
initial_state = selected_game.initial_state()

def selfplay(first_network_path, first_network_name, second_network_path, second_network_name, state_example):

  neural_network_1 = BreakthroughNN(state_example.cols, state_example.rows, state_example.get_move_amount())
  neural_network_2 = BreakthroughNN(state_example.cols, state_example.rows, state_example.get_move_amount())

  neural_network_1.loadmodel(first_network_path, first_network_name)

  initial_node = Node(state_example.initial_state(), "START")
  first_win = 0
  second_win = 0
  zero_action_sum = 0
  while True:


    for _ in tqdm(range(1000)):
    #   print("==========================")
    #   print("setting game to initial")
      curr_node = initial_node
    #   curr_node.gamestate.print_board()
    #   print("==========================")
      while True:
        # First NN moves
        pi,_ = neural_network_1.safe_predict(curr_node.gamestate)
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
          zero_action_sum += 1
        else:
          pi = pi / total

        curr_node = np.random.choice(curr_node.children, p=pi)

        if curr_node.gamestate.is_terminal():
          if curr_node.gamestate.reward() == 1:
            first_win += 1
          break

        # Second NN moves (is playing black)
        pi,_ = neural_network_2.safe_predict(curr_node.gamestate)
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
    print("Bestnetwork {} random {} ZERO ACTION SUM {}".format(first_win, second_win, zero_action_sum))

network_path = "./trained_models"
selfplay(network_path,"test1.tar", network_path, "test24.tar",initial_state)
