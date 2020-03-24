"""
generate dataset with heuristics
"""

from game_environments.breakthrough.breakthrough import BTBoard, config as BTconfig
import pandas as pd
import random

import numpy as np
from tqdm import tqdm

GAME = {
    "breakthrough": BTBoard(np.zeros([5,4]), 1),
}

selected_game = GAME["breakthrough"]
initial_state = selected_game.initial_state()

def generate_heuristic_dataset(game_amount:int):

  # breakthrough specific heuristics
  data = {
    "board_state":[],
    "player_piece_amount": [],
    "piece_difference":[],
    "furthest_piece":[],
    "furthest_piece_difference": []
  }

  for _ in tqdm(range(10)):
    curr_node = initial_state
    while not curr_node.is_terminal():
      curr_node.print_board()
      print(curr_node.get_heuristics())
      data["board_state"].append(curr_node.encode_state())
      for heuristic, val in curr_node.get_heuristics():
        data[heuristic].append(val)
      legal_moves = curr_node.legal_moves()
      curr_node = curr_node.execute_move(random.choice(legal_moves))

  return pd.DataFrame(data)

df = generate_heuristic_dataset(100)

df.to_csv("./heuristic_dataset_100_random.csv")