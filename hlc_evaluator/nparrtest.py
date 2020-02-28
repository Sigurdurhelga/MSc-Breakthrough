import numpy as np
from collections import namedtuple

gameconfig = namedtuple("gameconfig", "WHITE BLACK EMPTY")
config = gameconfig(-1,1,0)

def encode_state(board) -> np.ndarray:
    white = [[False for _ in range(3)] for _ in range(5)]
    black = [[False for _ in range(3)] for _ in range(5)]
    turn_white = True
    turn = [[turn_white for _ in range(3)] for _ in range(5)]
    for y in range(5):
        for x in range(3):
            if board[y,x] == config.WHITE:
                white[y][x] = True
            elif board[y,x] == config.BLACK:
                black[y][x] = True
    return np.array([white,black,turn])

a = np.array([[1,1,1],
     [0,0,0],
     [0,0,0],
     [0,0,0],
     [-1,-1,-1]])

print(encode_state(a))
