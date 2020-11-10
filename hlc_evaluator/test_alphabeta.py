from game_environments.breakthrough.breakthrough import BTBoard
from search_algorithms.alphabeta import alpha_beta_search
import numpy as np

state = BTBoard(np.zeros([6,6]), 1).initial_state()

while not state.is_terminal():
    state.print_board()
    state,_ = alpha_beta_search(state, 5, "player_piece_amount")
    if not state.is_terminal():
        state.print_board()
        state,_ = alpha_beta_search(state, 1, "player_piece_amount")

state.print_board()
    

