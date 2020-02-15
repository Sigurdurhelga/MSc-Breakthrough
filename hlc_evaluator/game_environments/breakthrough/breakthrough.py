import numpy as np
from copy import deepcopy, copy
import random
from collections import namedtuple

from game_environments.gamenode import GameNode

gameconfig = namedtuple("gameconfig", "WHITE BLACK EMPTY")
config = gameconfig(-1,1,0)


class BTBoard(GameNode):
    def __init__(self, board_list, player):
        self.board = board_list
        self.rows = len(board_list)
        self.cols = len(board_list[0])
        self.player = player
        self.terminal = self.is_terminal_helper()


    def legal_moves(self) -> list:
        d = self.player
        moves = []
        for y in range(self.rows):
            for x in range(self.cols):
                if self.board[y][x] != self.player:
                    continue
                if y + d >= self.rows or y + d < 0:
                    continue
                if x > 0:
                    if self.board[y+d][x-1] == (0-self.player):
                        moves.append((y,x,y+d,x-1))
                if x < self.cols-2:
                    if self.board[y+d][x+1] == (0-self.player):
                        moves.append((y,x,y+d,x+1))
                if self.board[y+d][x] == config.EMPTY:
                    moves.append((y,x,y+d,x))
        return moves

    def execute_move(self,move):
        new_board = np.copy(self.board)
        y1,x1,y2,x2 = move
        new_board[y2,x2] = new_board[y1,x1]
        new_board[y1,x1] = config.EMPTY
        return BTBoard(new_board, 0-self.player)

    def is_terminal(self) -> bool:
        return self.terminal

    def reward(self) -> int:
        if any(x == config.BLACK for x in self.board[-1,:]):
            return -1
        if any(x == config.WHITE for x in self.board[0,:]):
            return 1
        return 0

    def is_terminal_helper(self) -> bool:
        # print("calling is terminal helper on \n",self.board)
        # print("black on edge,",any(x == config.BLACK for x in self.board[-1]))
        # print("black on edge,",any(x == config.WHITE for x in self.board[0]))
        # print("legal moves,",self.legal_moves())
        return any(x == config.BLACK for x in self.board[-1,:]) \
                or any(x == config.WHITE for x in self.board[0,:]) \
                or len(self.legal_moves()) == 0

    def print_board(self):
        print("BOARD STATE")
        print("Turn: ", "White" if self.player == config.WHITE else "Black")
        print("Terminal:",self.terminal,end=" | ")
        if self.terminal:
            print("Winner:", "black" if any(x == config.BLACK for x in self.board[-1,:]) else "white" if any(x == config.WHITE for x in self.board[0,:]) else "tie")
        else:
            print()
        for row in self.board:
            print("".join(["w" if c == config.WHITE else "b" if c == config.BLACK else "·" for c in row]))
        print("------------------")
        print("legal moves:",self.legal_moves())
        print("==================")

    def __copy__(self):
        return np.copy(self.board)

    def __str__(self) -> str:
        return "|".join(["".join(["w" if c == config.WHITE else "b" if c == config.BLACK else " " for c in row]) for row in self.board])+"|{}".format(self.player)

    def __hash__(self) -> int:
        return str.__hash__(self.__str__())

    def __eq__(self, other) -> bool:
        return self.board == other.board


