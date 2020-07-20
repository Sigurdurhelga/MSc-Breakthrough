from flask import Flask, render_template, url_for, send_from_directory, request, jsonify
from flask_cors import CORS
import numpy as np
from game_environments.breakthrough.breakthrough import BTBoard

import json

app = Flask(__name__, static_folder='static')
CORS(app)

"""
@app.route('/')
def home():
    return render_template('home.html')
"""

working_board = BTBoard(np.zeros([6,6]), 1).initial_state()

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
    working_board = working_board.initial_state()
    to_ret = board_to_jsonifyable(working_board)
    return jsonify(to_ret)

@app.route('/post_move', methods=['POST'])
def post_move():
    global working_board
    data = request.json
    y1,x1 = int(data['from'][3]), int(data['from'][1])
    y2,x2 = int(data['to'][3]) , int(data['to'][1])
    move = (y1,x1,y2,x2)
    if move in working_board.legal_moves:
        working_board = working_board.execute_move(move)
        to_ret = board_to_jsonifyable(working_board)
        to_ret['valid'] = True
        to_ret['terminal'] = working_board.is_terminal()
        return jsonify(to_ret)
    else:
        return jsonify({"valid":False})


if __name__ == '__main__':
    app.run(debug=True)
