"""
  nn_pit module

  This module focuses on pitting two neural networks against one another,
  Firstly a randomly nn will start playing, will generate data, then use
  that data to learn. After that the NN will play against it's previous
  iteration, the one who wins will be the new primary and will generate
  data and train itself.
"""

from monte_carlo.mcts import MCTS
from monte_carlo.mctsnode import Node
from game_environments.play_chess.playchess import PlayChess
from game_environments.gamenode import GameNode
from game_environments.breakthrough.breakthrough import BTBoard, config as BTconfig
from neural_networks.breakthrough.breakthrough_nn import BreakthroughNN

import numpy as np
from tqdm import tqdm


GAME = {
    "breakthrough": BTBoard(np.zeros([7,6]), 1),
}

selected_game = GAME["breakthrough"]
initial_state = selected_game.initial_state()

EPISODE_AMOUNT = 2

NEURAL_NETWORK_THINK = 2

TEMP_THRESHOLD = 10

def generate_dataset(primary_nn: BreakthroughNN, initial_state : GameNode, verbose=False):
  global NEURAL_NETWORK_THINK
  initial_node = Node(initial_state,"START")
  monte_tree = MCTS(initial_node, primary_nn)
  dataset = []


  for _ in tqdm(range(EPISODE_AMOUNT)):
    monte_tree.set_node(monte_tree.initial_root)
    curr_node = monte_tree.root
    action_depth = 0
    episode_data = []
    while True:
      if verbose:
        print("=====================")
        print("ACTION DEPTH:",action_depth)
        curr_node.gamestate.print_board()
        print("=====================")
      p,_ = primary_nn.predict(curr_node.gamestate)
      datapoint = [curr_node.gamestate, p, 0]
      episode_data.append(datapoint)
      if curr_node.gamestate.is_terminal():
        break
      # selection / expantion / rollout
      monte_tree.nn_rollout(NEURAL_NETWORK_THINK)
      # NNpolicy based select select best
      temp = 1 if action_depth < TEMP_THRESHOLD else 0
      pi = monte_tree.get_policy(temp)
      best_child = np.random.choice(curr_node.children, p=pi)
      monte_tree.move_to_child(best_child.action)
      curr_node = monte_tree.root
      action_depth += 1
    reward = curr_node.gamestate.reward()
    for i in range(len(episode_data)-1):
      episode_data[i+1][2] = reward
    dataset.extend(episode_data)
  return dataset

first_nn = BreakthroughNN(initial_state.cols, initial_state.rows, initial_state.get_move_amount())
print(generate_dataset(first_nn, initial_state,True))