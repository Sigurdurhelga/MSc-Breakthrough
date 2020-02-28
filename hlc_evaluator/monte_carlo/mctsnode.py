from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
from game_environments.gamenode import GameNode

class Node():
  """
    Class reperesenting nodes within a MCTS tree
  """
  def __init__(self, gamestate: GameNode, action=""):
    self.gamestate = gamestate
    self.action = action
    self.expanded = False
    self.children = []
    self.pidx = self.action_idx_to_policy_idx()

  def get_pidx(self) -> int:
    return self.pidx

  def expand(self) -> None:
    "fills the children list in the node"
    assert not self.is_expanded(), "calling expand on expanded node is bad"
    self.children = [None] * self.gamestate.get_move_amount()
    for move in self.gamestate.legal_moves():
      new_child = Node(self.gamestate.execute_move(move), move)
      self.children[new_child.get_pidx()] = new_child
    self.expanded = True

  def action_idx_to_policy_idx(self) -> int:
    if self.action == "START":
      return
    x1,y1,x2,y2 = self.action
    direction = 0 if y1 < y2 else 3
    direction += 0 if x1 < x2 else 1 if x1 == x2 else 2
    # the 6 here is breakthrough specific make a method
    return x2 + ( self.gamestate.cols * ( y2 + ( 6 * direction ) ) )

  def is_expanded(self) -> bool:
    return self.expanded

  def __hash__(self) -> int:
    return hash(self.gamestate)+hash(self.action)

  def __eq__(self, other) -> bool:
    return str(self.gamestate) == str(other.gamestate)

