
"""
function negamax(node, depth, α, β, color) is
    if depth = 0 or node is a terminal node then
        return color × the heuristic value of node

    childNodes := generateMoves(node)
    childNodes := orderMoves(childNodes)
    value := −∞
    foreach child in childNodes do
        value := max(value, −negamax(child, depth − 1, −β, −α, −color))
        α := max(α, value)
        if α ≥ β then
            break (* cut-off *)
    return value
"""

from game_environments.gamenode import GameNode
from game_environments.breakthrough.breakthrough import config

"""
all_heuristics = [
    "player_piece_amount",
    "piece_difference",
    "furthest_piece",
    "furthest_piece_difference"
]
"""

selected_heuristic = None

def alphabeta(node, depth, alpha, beta):
    global selected_heuristic
    if depth == 0 or node.is_terminal():
        # special case as my gamenodes always give their heuristics as positive from their viewpoint
        heuristic = node.get_heuristics(selected_heuristic)
        if node.player == config.BLACK:
            heuristic *= -1
        return heuristic

    child_nodes = [node.execute_move(move) for move in node.legal_moves]
    val = -float("inf")

    for node in child_nodes:
        val = max(val, -alphabeta(node, depth - 1, -beta, -alpha))
        alpha = max(alpha,val)
        if alpha >= beta:
            break
    return val

def alpha_beta_search(node,depth,heuristic):
    global selected_heuristic
    selected_heuristic = heuristic
    children = [node.execute_move(move) for move in node.legal_moves]
    is_maxing = node.player == config.WHITE
    best_val = -float("inf") if is_maxing else float("inf")
    best_child = None
    for child in children:
        child_val = alphabeta(child,depth-1,-float("inf"),float("inf"))
        if is_maxing and child_val > best_val:
            best_val = child_val
            best_child = child
        elif not is_maxing and child_val < best_val:
            best_val = child_val
            best_child = child
    return best_child
        



