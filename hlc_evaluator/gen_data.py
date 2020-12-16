import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from game_environments.breakthrough.breakthrough import BTBoard, config as BTconfig
import random
from tqdm import tqdm
from PIL import Image
import ast
from itertools import permutations
from monte_carlo.mcts import MCTS
from monte_carlo.mctsnode import Node

def from_np_array(array_string):
    array_string = ','.join(array_string.replace('[ ', '[').split())
    return np.array(ast.literal_eval(array_string))

GAME = {
    "breakthrough": BTBoard(np.zeros([6,6]), 1),
}
selected_game = GAME["breakthrough"]
initial_state = selected_game.initial_state()
seen_states = {}

frontier = [initial_state]

generate_new_data = True
new_data_size = 1000000
dfs = True

if generate_new_data:
    df = pd.DataFrame(columns=["id",
                               "player",
                               "state",
                               "terminal", 
                               "player_piece_amount", 
                               "piece_difference", 
                               "furthest_piece", 
                               "furthest_piece_difference"], 
                      dtype=np.int64
                      )
    state_id = 0

    while frontier and state_id < new_data_size:
        working_state = frontier.pop() if dfs else frontier.pop(0)
        if working_state.__hash__() in seen_states:
            continue


        heuristics = working_state.get_heuristics()
        df = df.append({
            'id':state_id,
            'state':working_state.encode_state(),
            'player':min(working_state.player+1,1),
            'terminal':working_state.is_terminal(),
            'player_piece_amount':heuristics[0][1],
            'piece_difference':heuristics[1][1],
            'furthest_piece':heuristics[2][1],
            'furthest_piece_difference':heuristics[3][1]
        },ignore_index=True)
        
        state_id += 1
        seen_states[working_state.__hash__()] = True
        np.random.shuffle(working_state.legal_moves)
        for lm in working_state.legal_moves:
            frontier.append(working_state.execute_move(lm))
    df.set_index('id') 

    df.to_csv(f'./{"dfs" if dfs else "bfs"}_moves_heuristic_{new_data_size}.csv')
else:
    df = pd.read_csv(f'./{"dfs" if dfs else "bfs"}_moves_heuristic_{new_data_size}.csv', converters={'state': from_np_array}).set_index('id')
