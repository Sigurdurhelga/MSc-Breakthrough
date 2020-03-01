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
    "breakthrough": BTBoard(np.zeros([5,4]), 1),
}

selected_game = GAME["breakthrough"]
initial_state = selected_game.initial_state()


"""
  Config varibles
"""
EPISODE_AMOUNT = 2
NEURAL_NETWORK_THINK = 50
TEMP_THRESHOLD = 6
TRAINING_ITERS = 3

def generate_dataset(primary_nn: BreakthroughNN, game_example : GameNode, saved_monte_tree=None, verbose=False):
  global NEURAL_NETWORK_THINK

  initial_node = Node(game_example.initial_state(), "START")

  if not saved_monte_tree:
    monte_tree = MCTS(initial_node, primary_nn)
  else:
    monte_tree = saved_monte_tree

  dataset = []

  for _ in tqdm(range(EPISODE_AMOUNT)):
    monte_tree.set_node(monte_tree.initial_root)
    curr_node = monte_tree.root
    action_depth = 0
    episode_data = []
    while True:
      if curr_node.gamestate.is_terminal():
        break

      # selection / expantion / rollout
      monte_tree.nn_rollout(NEURAL_NETWORK_THINK)

      # NNpolicy based select select best
      temp = 1 if action_depth < TEMP_THRESHOLD else 0
      pi = monte_tree.get_policy(temp)
      datapoint = [curr_node.gamestate.encode_state(), pi, 0]
      episode_data.append(datapoint)
      best_child = np.random.choice(curr_node.children, p=pi)
      monte_tree.move_to_child(best_child.action)
      curr_node = monte_tree.root
      action_depth += 1

    reward = curr_node.gamestate.reward()
    episode_data.append([curr_node.gamestate.encode_state(), monte_tree.get_policy(),0])

    for i in range(len(episode_data)-1):
      episode_data[i+1][2] = reward
    dataset.extend(episode_data)
  print("[TRAINING] datapoints gathered amount: {}".format(len(dataset)))

  return dataset

def train_model(play_iterations, neural_network: BreakthroughNN, state_example: GameNode):
  for _ in tqdm(range(play_iterations)):
    dataset = generate_dataset(neural_network, state_example)
    neural_network.train(dataset)



def selfplay(first_network_path, first_network_name, second_network_path, second_network_name, state_example):

  neural_network_1 = BreakthroughNN(state_example.cols, state_example.rows, state_example.get_move_amount())
  neural_network_2 = BreakthroughNN(state_example.cols, state_example.rows, state_example.get_move_amount())

  neural_network_1.loadmodel(first_network_path, first_network_name)
  neural_network_2.loadmodel(second_network_path, second_network_name)

  initial_node = Node(state_example.initial_state(), "START")

  generation = 1

  while True:

    first_win = 0
    second_win = 0

    for _ in tqdm(range(100)):
      curr_node = initial_node
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
    print("[trainer.py] Episode record was White: {} | Black: {} | Tie: {}".format(first_win, second_win, 10 - (first_win+second_win)))
    if first_win > second_win:
      print("[trainer.py] First network better than second network, saving first")
      neural_network_2.loadmodel(first_network_path, first_network_name)
      neural_network_1.savemodel("./trained_models", "best_network.tar")
    elif second_win > first_win:
      print("[trainer.py] Second network better than first network, saving second")
      neural_network_1.loadmodel(second_network_path, second_network_name)
      neural_network_1.savemodel("./trained_models", "best_network.tar")

    neural_network_1.savemodel(first_network_path,first_network_name)
    neural_network_2.savemodel(second_network_path,second_network_name)
    print("[trainer.py] STARTING TRAINING")
    train_model(TRAINING_ITERS, neural_network_1, state_example)
    print("[trainer.py] DONE TRAINING")
    print("[trainer.py] GENERATION {}".format(generation))
    generation += 1

network_path = "./trained_models"
selfplay(network_path,"working1.tar", network_path, "working2.tar",initial_state)






