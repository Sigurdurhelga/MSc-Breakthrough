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


"""
  Config varibles
"""
EPISODE_AMOUNT = 2
NEURAL_NETWORK_THINK = 100
TEMP_THRESHOLD = 10000
TRAINING_ITERS = 10
VERIFICATION_GAMES = 10

ITERATION = 0

def generate_dataset(primary_nn: BreakthroughNN, game_example : GameNode, saved_monte_tree=None, verbose=False):
  global ITERATION
  curr_node = Node(game_example.initial_state(), "START")
  monte_tree = MCTS()
  dataset = []


  temp = 1 * (0.999 ** (ITERATION))
  print("[trainer.py] playing with temp:",temp)

  whiteplaying = True

  # firstly generate dataset from white point of view
  while not curr_node.gamestate.is_terminal():

    # NNpolicy based select select best
    if temp < 0.1:
      temp = 0

    pi = monte_tree.get_policy(curr_node,NEURAL_NETWORK_THINK, primary_nn, temp)
    for i,child in enumerate(curr_node.children):
      if not child:
        pi[i] = 0
    pi = pi / sum(pi)

    # if whiteplaying:
    datapoint = [curr_node.gamestate.encode_state(), pi, 0]
    dataset.append(datapoint)

    curr_node = np.random.choice(curr_node.children, p=pi)

  reward = curr_node.gamestate.reward()
  print("[generating dataset] playing as white reward {}".format(reward))

  for i in range(len(dataset)-1):
    dataset[i+1][2] = reward
  # print("[TRAINING] datapoints gathered amount: {}".format(len(dataset)))
  monte_tree = MCTS()

  curr_node = Node(game_example.initial_state(), "START")
  # secondly generate dataset from blacks point of view
  black_dataset = []
  while not curr_node.gamestate.is_terminal():

    # NNpolicy based select select best
    if temp < 0.1:
      temp = 0

    pi = monte_tree.get_policy(curr_node,NEURAL_NETWORK_THINK,primary_nn,temp)
    for i,child in enumerate(curr_node.children):
      if not child:
        pi[i] = 0
    pi = pi / sum(pi)

    datapoint = [curr_node.gamestate.encode_state(), pi, 0]
    black_dataset.append(datapoint)

    curr_node = np.random.choice(curr_node.children, p=pi)

  reward = curr_node.gamestate.reward()
  print("[generating dataset] playing as black reward {}".format(reward))

  for i in range(len(black_dataset)-1):
    black_dataset[i+1][2] = reward

  dataset.extend(black_dataset)

  return dataset

def train_model(neural_network: BreakthroughNN, state_example: GameNode):
  for _ in tqdm(range(TRAINING_ITERS)):
    for _ in tqdm(range(EPISODE_AMOUNT)):
      dataset = generate_dataset(neural_network, state_example)
      neural_network.train(dataset)



def selfplay(first_network_path, first_network_name, second_network_path, second_network_name, state_example):
  global ITERATION

  neural_network_1 = BreakthroughNN(state_example.cols, state_example.rows, state_example.get_move_amount())
  neural_network_2 = BreakthroughNN(state_example.cols, state_example.rows, state_example.get_move_amount())

  neural_network_1.loadmodel(first_network_path, first_network_name)
  neural_network_2.loadmodel(second_network_path, second_network_name)

  initial_node = Node(state_example.initial_state(), "START")

  generation = 1

  while True:
    ITERATION += 1
    first_win = 0
    second_win = 0
    if generation != 1:
      print("[trainer.py] STARTING VERIFICATION OF NN's")

      for _ in tqdm(range(VERIFICATION_GAMES)):
        monte_tree_1 = MCTS()
        monte_tree_2 = MCTS()
        curr_node = initial_node
        while True:
          # WHITE MOVES
          pi = monte_tree_1.get_policy(curr_node, NEURAL_NETWORK_THINK, neural_network_1)

          for i,child in enumerate(curr_node.children):
            if not child:
              pi[i] = 0
          pi = pi / sum(pi)

          curr_node = np.random.choice(curr_node.children, p=pi)

          if curr_node.gamestate.is_terminal():
            if curr_node.gamestate.reward() == 1:
              first_win += 1
            break

          # BLACK MOVES
          pi = monte_tree_2.get_policy(curr_node, NEURAL_NETWORK_THINK, neural_network_2)

          for i,child in enumerate(curr_node.children):
            if not child:
              pi[i] = 0
          pi = pi / sum(pi)

          curr_node = np.random.choice(curr_node.children, p=pi)

          if curr_node.gamestate.is_terminal():
            if curr_node.gamestate.reward() == -1:
              second_win += 1
            break

      print("[trainer.py] Episode record was White: {} | Black: {} | Tie: {}".format(first_win, second_win, VERIFICATION_GAMES - (first_win+second_win)))
      print("[trainer.py] White winrate {} | black winrate {}".format((first_win/VERIFICATION_GAMES),(second_win/VERIFICATION_GAMES)))
      if (first_win/VERIFICATION_GAMES) > 0.55:
        print("[trainer.py] First network wins, saving first")
        neural_network_2.loadmodel(first_network_path, first_network_name)

        neural_network_1.savemodel("./trained_models", "best_network.tar")
        neural_network_1.savemodel(first_network_path,first_network_name)
        neural_network_2.savemodel(second_network_path,second_network_name)
      elif (second_win/VERIFICATION_GAMES) > 0.55:
        print("[trainer.py] Second network wins, saving second")
        neural_network_1.loadmodel(second_network_path, second_network_name)

        neural_network_1.savemodel("./trained_models", "best_network.tar")
        neural_network_1.savemodel(first_network_path,first_network_name)
        neural_network_2.savemodel(second_network_path,second_network_name)

    print("[trainer.py] STARTING TRAINING")
    train_model(neural_network_1, state_example)
    print("[trainer.py] DONE TRAINING")
    print("[trainer.py] GENERATION {}".format(generation))
    generation += 1

if __name__ == "__main__":
  network_path = "./trained_models"
  selfplay(network_path,"working1.tar", network_path, "working2.tar",initial_state)






