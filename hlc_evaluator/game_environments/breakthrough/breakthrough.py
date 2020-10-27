
import numpy as np
from copy import deepcopy, copy
import random
from collections import namedtuple

from game_environments.gamenode import GameNode

gameconfig = namedtuple("gameconfig", "WHITE BLACK EMPTY")
config = gameconfig(-1,1,0)

LEGAL_MOVES_CACHE = {}


class BTBoard(GameNode):
    """
        BTBoard is the game class for a game of breakthrough

        Member variables:
            board:   np.array - represents the current game of breakthrough
            row:     int      - the height of the game board
            cols:    int      - the width of the game board
            player:  int      - 1: white player playing -1: black player playing
            terminal:bool     - true if current state is a terminal state
    """

    def __init__(self, board_list, player, terminal=False,white_count=-1,black_count=-1):
        super(BTBoard, self).__init__()
        self.board = board_list
        self.rows = len(board_list)
        self.cols = len(board_list[0])
        self.player = player
        if white_count == -1 and black_count == -1:
            self.white_count = self.cols*2
            self.black_count = self.cols*2
        else:
            self.white_count = white_count
            self.black_count = black_count
        self.stringified = ""
        self.legal_moves = []
        self.terminal = terminal
        if not self.terminal:
            self.legal_moves_helper()
        if self.white_count == 0 or self.black_count == 0:
            self.terminal = True
    
    def legal_moves_helper(self):
        """
        Legal_moves()
            - sets self.legal_moves to a list of legal moves for the BTBoard
                list contains tuples with (x1, y1, x2, y2) representing
                moving piece at col x1 row y2 to col x2 row y2
        """
        # global LEGAL_MOVES_CACHE
        # if self.__str__() in LEGAL_MOVES_CACHE:
            # self.legal_moves = LEGAL_MOVES_CACHE[self.__str__()]
        # else:
        d = self.player
        moves = []
        for y in range(self.rows):
            for x in range(self.cols):
                if self.board[y][x] != d:
                    continue
                elif y + d >= self.rows or y + d < 0:
                    continue
                # move to the left
                if x > 0 and self.board[y+d][x-1] != d:
                    moves.append((y,x,y+d,x-1))
                # move to the right
                if x < self.cols-1 and self.board[y+d][x+1] != d:
                    moves.append((y,x,y+d,x+1))
                # move straight
                if self.board[y+d][x] == config.EMPTY:
                    moves.append((y,x,y+d,x))

        # LEGAL_MOVES_CACHE[self.__str__()] = moves
        self.legal_moves = moves

    def execute_move(self,move):
        """
        Execute_move(move)
            - Returns a new BTBoard representing the gamestate
              after applying move to the (self) BTBoard
        """
        new_board = deepcopy(self.board)
        y1,x1,y2,x2 = move

        other_player = (0 - (self.player))
        other_count = self.white_count if self.player == config.BLACK else self.white_count
        kill_move = self.board[y2,x2] == other_player
        white_count = self.white_count
        black_count = self.black_count 
        if kill_move and self.player == config.WHITE:
            black_count -= 1
        elif kill_move and self.player == config.BLACK:
            white_count -= 1

        new_board[y2,x2] = new_board[y1,x1]
        new_board[y1,x1] = config.EMPTY
        is_terminal = False
        if self.player == config.WHITE and y2 == 0:
            is_terminal = True
        elif self.player == config.BLACK and y2 == self.rows-1:
            is_terminal = True
        elif kill_move and other_count == 1:
            is_terminal = True

        return BTBoard(new_board, 0-self.player, terminal=is_terminal, white_count=white_count, black_count=black_count)

    def is_terminal(self) -> bool:
        return self.terminal

    def reward(self) -> int:
        """
        reward()
            - returns 1 if white wins, -1 if black wins, and 0 on a draw
        """
        assert self.is_terminal()
        if any(x == config.BLACK for x in self.board[-1,:]):
            return -1
        if any(x == config.WHITE for x in self.board[0,:]):
            return 1

        return 0 - (self.player)

    def is_terminal_helper(self) -> bool:
        return any(x == config.BLACK for x in self.board[-1,:]) or any(x == config.WHITE for x in self.board[0,:])

    def initial_state(self):
        new_board = np.zeros([self.rows, self.cols]).astype(int)
        new_board[:2,:] = config.BLACK
        new_board[-2:,:] = config.WHITE
        return BTBoard(new_board, config.WHITE)

    def encode_state(self) -> np.ndarray:
        """
        encode_state()
            - Returns an encoded version of the BTBoard, this encoded
              version will be run through NeuralNetworks
        """
        enc_board = np.zeros([3,self.rows, self.cols])
        for y in range(self.rows):
            for x in range(self.cols):
                if self.board[y,x] == config.WHITE:
                    enc_board[0,y,x] = 1
                elif self.board[y,x] == config.BLACK:
                    enc_board[1,y,x] = 1
        if self.player == config.WHITE:
            enc_board[2,:,:] = 1
        return enc_board

    def print_board(self):
        print("BOARD STATE")
        print("Turn: ", "White" if self.player == config.WHITE else "Black")
        print("Terminal:",self.terminal,end=" | ")
        if self.terminal:
            print("Winner:", "black" if self.player == config.WHITE else "white")
        else:
            print()
        print("-"*(self.cols + 4 + self.cols - 1))
        for idx,row in enumerate(self.board):
            print(str(idx)+" "+" ".join(["w" if c == config.WHITE else "b" if c == config.BLACK else "·" for c in row]))
        print("  "+" ".join([str(x) for x in list(range(self.cols))]))
        print("-"*(self.cols + 4 + self.cols - 1))
        print("legal moves:",self.legal_moves)
        print("==================")

    """
        HEURISTICS START
    """

    def heuristic_player_piece_amount(self):
        count = 0
        for y in range(self.rows):
            for x in range(self.cols):
                if self.board[y,x] == self.player:
                    count += 1
        return count

    def heuristic_piece_difference(self):
        count = 0
        for y in range(self.rows):
            for x in range(self.cols):
                count += self.board[y,x]
        if self.player == config.WHITE:
            count *= -1
        return count


    def heuristic_furthest_piece(self):
        if self.is_terminal():
            return 0
        if self.player == config.WHITE:
            for y in range(self.rows):
                for x in range(self.cols):
                    if self.board[y,x] == config.WHITE:
                        return self.rows - y - 1
        else:
            for y in range(self.rows-1, -1, -1):
                for x in range(self.cols):
                    if self.board[y,x] == config.BLACK:
                        return y
        return -1

    def heuristic_furthest_piece_difference(self):
        furthest_white = 0
        furthest_black = 0
        found_white = False
        found_black = False
        for y in range(self.rows):
            for x in range(self.cols):
                if self.board[y,x] == config.WHITE:
                    furthest_white = self.rows - y - 1
                    found_white = True
                    break
            if found_white:
                break
        for y in range(self.rows-1,-1,-1):
            for x in range(self.cols):
                if self.board[y][x] == config.BLACK:
                    furthest_black = y
                    found_black = True
                    break
            if found_black:
                break
        out = furthest_white - furthest_black
        if self.player == config.BLACK:
            out = -out
        return out

    def heuristic_average_distance(self):
        pawn_heights = []
        if self.player == config.WHITE:
            for y in range(self.rows):
                for x in range(self.cols):
                    if self.board[y,x] == self.player:
                        pawn_heights.append((self.rows-1) - y)
        else:
            for y in range(self.rows):
                for x in range(self.cols):
                    if self.board[y,x] == self.player:
                        pawn_heights.append(y)
        lowest_pawn = min(pawn_heights)
        highest_pawn = max(pawn_heights)
        middle_point = (highest_pawn + lowest_pawn) / 2
        return sum([p - middle_point for p in pawn_heights]) / len(pawn_heights)

    def heuristic_lorentz_horey(self):
        """
        position values for black, reverse for white
        """
        position_values = np.array([
            [ 5,15, 5, 5,15, 5],
            [ 2, 3, 3, 3, 3, 2],
            [ 4, 6, 6, 6, 6, 4],
            [ 7,10,10,10,10, 7],
            [ 7,10,10,10,10, 7],
            [15,15,15,15,15,15],
        ])

        if self.player == config.WHITE:
            position_values = position_values[::-1]
        total = 0
        for y in range(self.rows):
            for x in range(self.cols):
                if self.board[y,x] == self.player:
                    total += position_values[y,x]
        return total
    def heuristic_lorentz_horey_difference(self):
        """
        position values for black, reverse for white
        """
        black_position_values = np.array([
            [ 5,15, 5, 5,15, 5],
            [ 2, 3, 3, 3, 3, 2],
            [ 4, 6, 6, 6, 6, 4],
            [ 7,10,10,10,10, 7],
            [ 11,15,15,15,15, 11],
            [21,21,21,21,21,21],
        ])
        white_position_values = black_position_values[::-1]

        total_black = 0
        total_white = 0
        for y in range(self.rows):
            for x in range(self.cols):
                if self.board[y,x] == config.WHITE:
                    total_white += white_position_values[y,x]
                if self.board[y,x] == config.BLACK:
                    total_black += black_position_values[y,x]
        
        return total_black - total_white if self.player == config.BLACK else total_white - total_black




    def get_heuristics(self, name=""):
        all_heuristics = {
            "player_piece_amount":self.heuristic_player_piece_amount,
            "piece_difference":self.heuristic_piece_difference,
            "furthest_piece": self.heuristic_furthest_piece,
            "furthest_piece_difference":self.heuristic_furthest_piece_difference,
            "average_distance":self.heuristic_average_distance,
            "lorentz_horey": self.heuristic_lorentz_horey,
            "lorentz_horey_difference": self.heuristic_lorentz_horey_difference
        }

        results = []
        # calculcate a specific heuristic
        if name:
            return all_heuristics[name]()
        else:
            for k,v in all_heuristics.items():
                results.append((k, v()))
        return results

    """
        HEURISTICS END
    """

    def get_move_amount(self):
        return self.rows * self.cols * 6 # representing all directions we can move

    def __copy__(self):
        return np.copy(self.board)

    def __str__(self) -> str:
        if self.stringified == "":
            self.stringified = "|".join(["".join([str(c) for c in row]) for row in self.board])+"|{}|{}".format(self.player,self.terminal)
        return self.stringified

    def __hash__(self) -> int:
        return str.__hash__(self.__str__())

    def __eq__(self, other) -> bool:
        return self.__str__() == other.__str__()

    def __ne__(self,other) -> bool:
        return self.__str__() != other.__str__()


