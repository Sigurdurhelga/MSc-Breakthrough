import chess
import copy
from mctsnode import Node
import random

class PlayChess(Node):

  rewards_dict = {
    '1/2-1/2': 0,
    '1-0': 1,
    '0-1': -1
  }

  def __init__(self, fen=""):
    if fen:
      self.board = chess.Board(fen)
    else:
      self.board = chess.Board()
    self.white_turn = self.board.turn
    self.terminal = self.board.is_game_over(claim_draw=True)
    self.moves = list(self.board.legal_moves)

  def is_terminal(self) -> bool:
    return self.terminal

  def reward(self) -> int:
    assert(not self.is_terminal(), "calling reward on non terminal node is not allowed")
    return self.rewards_dict[self.board.result(claim_draw=True)]

  def find_random_child(self):
    if self.is_terminal():
      return None
    self.board.push(random.choice(self.moves))
    new_board = PlayChess(self.board.fen())
    self.board.pop()
    return new_board

  def get_children(self) -> set:
    children = set()
    for m in self.moves:
      self.board.push(m)
      children.add(PlayChess(self.board.fen()))
      self.board.pop()
    return children

  def make_move(self, move: chess.Move):
    self.board.push(move)
    new_board = copy.deepcopy(self.board)
    self.board.pop()
    return PlayChess(new_board.fen())

  def print_board(self) -> None:
    print(self.board)


  """
    The magic functions utilized just abuse the string magic functions
    assuming they work perfectly... it should be fine.
  """
  def __str__(self):
    return self.board.fen()

  def __hash__(self):
    return str.__hash__(self.board.fen())

  def __eq__(self, other):
    return str(self) == str(other)

