import numpy as np
from monte_carlo.mctsnode import Node
from copy import deepcopy
import random

BREAKTHROUGHWHITE = 1
BREAKTHROUGHBLACK = 2
BREAKTHROUGHEMPTY = 0


class BTBoard(Node):
    def __init__(self, board_list, player_turn):
        self.board = board_list
        self.rows = len(board_list)
        self.cols = len(board_list[0])
        self.turn = player_turn
        self.terminal = self.is_terminal_helper()

    def print_board(self):
        print("BOARD STATE")
        print("Turn: ", "White" if self.turn == BREAKTHROUGHWHITE else "Black")
        for row in self.board:
            print("".join([str[c] for c in row]))
        print("==================")

    def copy_board(self):
        return deepcopy(self.board)

    def find_random_child(self):
        return random.Choice(self.children)

    def gen_move(self, fpos, tpos):
        fposy, fposx = fpos
        tposy, tposx = tpos
        new_board = self.copy_board()
        new_board[tposy][tposx] = new_board[fposy][fposx]
        new_board[fposy][fposx] = BREAKTHROUGHEMPTY
        next_player = BREAKTHROUGHWHITE if self.turn == BREAKTHROUGHBLACK else BREAKTHROUGHBLACK
        b_new_board = BTBoard(new_board, next_player)
        return b_new_board

    def get_children(self) -> set:
        children = set()
        for y in range(self.rows - 1, 0, -1):  # only go up to second to last
            for x in range(self.cols):
                if self.board[y][x] == self.turn:
                    for d in self.get_dirs((y, x)):
                        children.add(self.gen_move((y, x), d))
        return children

    def get_dirs(self, pos) -> list:
        "pos is coordinates in list [y,x]"
        dirs = []
        do_left = pos[1] > 0
        do_right = pos[1] < (self.cols - 1)
        print("getdirs with ", pos)
        ymod = -1 if self.turn == BREAKTHROUGHWHITE else 1  # y axis mod
        enemy = BREAKTHROUGHWHITE if self.turn == BREAKTHROUGHBLACK else BREAKTHROUGHBLACK
        if do_left and self.board[pos[0] + ymod][pos[1] - 1] == enemy:
            dirs.append([pos[0] + ymod, pos[1] - 1])
        if do_right and self.board[pos[0] + ymod][pos[1] + 1] == enemy:
            dirs.append([pos[0] + ymod, pos[1] + 1])
        if self.board[pos[0] + ymod][pos[1]] == BREAKTHROUGHEMPTY:
            dirs.append([pos[0] + ymod, pos[1]])
        return dirs

    def is_terminal_helper(self) -> bool:
        return any(x == 2 for x in self.board[0]) \
               or any(x == 1 for x in self.board[-1]) \
               or len(self.legal_moves()) == 0

    def is_terminal(self) -> bool:
        return self.terminal

    def reward(self) -> int:
        assert self.is_terminal()
        if any(x == 2 for x in self.board[0]):
            return 1
        if any(x == 1 for x in self.board[-1]):
            return -1
        return 0

    def __str__(self) -> str:
        return "|".join(["".join([str(c) for c in row]) for row in self.board])

    def __hash__(self) -> int:
        return str.__hash__(self.__str__())

    def __eq__(self, other) -> bool:
        return self.board == other.board


