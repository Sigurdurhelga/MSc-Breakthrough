from flask import Flask, render_template, url_for, send_from_directory, request, jsonify
from flask_cors import CORS
import numpy as np
from game_environments.breakthrough.breakthrough import BTBoard

from monte_carlo.mcts import MCTS
from monte_carlo.mctsnode import Node
# from game_environments.play_chess.playchess import PlayChess
from game_environments.gamenode import GameNode
from neural_networks.breakthrough.breakthrough_nn import BreakthroughNN

import json

app = Flask(__name__, static_folder='static')
CORS(app)

"""
@app.route('/')
def home():
    return render_template('home.html')
"""

working_board = Node(BTBoard(np.zeros([6,6]), 1).initial_state(),"START")

def board_to_jsonifyable(bt_board):
    board_dict = {}
    board = bt_board.board.tolist()
    for y in range(len(board)):
        for x in range(len(board[0])):
            board_dict["x{}y{}".format(x,y)] = board[y][x]

    for y1,x1,y2,x2 in bt_board.legal_moves:
        key = f"mx{x1}y{y1}"
        if key in board_dict:
            board_dict[key].append(f"x{x2}y{y2}")
        else:
            board_dict[key] = [f"x{x2}y{y2}"]
    return board_dict


@app.route('/')
def breakthrough():
    blackimage = url_for('static', filename='images/black.png')
    whiteimage = url_for('static', filename='images/white.png')
    return render_template('breakthrough.html', white=whiteimage, black=blackimage)


@app.route('/get_board', methods=['GET'])
def get_board():
    global working_board
    working_board = Node(working_board.gamestate.initial_state(),"START")
    to_ret = board_to_jsonifyable(working_board.gamestate)
    return jsonify(to_ret)

@app.route('/post_move', methods=['POST'])
def post_move():
    global working_board
    data = request.json
    y1,x1 = int(data['from'][3]), int(data['from'][1])
    y2,x2 = int(data['to'][3]) , int(data['to'][1])
    move = (y1,x1,y2,x2)
    if move in working_board.gamestate.legal_moves:
        working_board = Node(working_board.gamestate.execute_move(move),move)
        to_ret = board_to_jsonifyable(working_board.gamestate)
        to_ret['valid'] = True
        to_ret['terminal'] = working_board.gamestate.is_terminal()
        return jsonify(to_ret)
    else:
        return jsonify({"valid":False})

neural_network = BreakthroughNN(working_board.gamestate.cols, working_board.gamestate.rows, working_board.gamestate.get_move_amount())
neural_network.loadmodel('./trained_models', 'session2_res5_gen300.tar')

@app.route('/get_ai_move', methods=['GET'])
def get_ai_move():
    global neural_network, working_board
    think = request.args.get('think', default=10, type=int)
    monte_tree = MCTS()
    pi,v = neural_network.safe_predict(working_board.gamestate)
    pi = pi.detach().cpu().numpy()[0]
    if not working_board.is_expanded():
        working_board.expand()
    for i,child in enumerate(working_board.children):
        if not child:
            pi[i] = 0

    pi = pi / sum(pi)

    working_board = np.random.choice(working_board.children, p=pi)

    y1,x1,y2,x2 = working_board.action
    to_ret = board_to_jsonifyable(working_board.gamestate)
    to_ret['from'] = f'x{x1}y{y1}'
    to_ret['to'] = f'x{x2}y{y2}'
    to_ret['terminal'] = working_board.gamestate.is_terminal()

    return jsonify(to_ret)

    



if __name__ == '__main__':
    app.run(debug=True)
