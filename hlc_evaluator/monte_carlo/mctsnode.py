from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
from game_environments.gamenode import GameNode

class Node():
  """
    Generic class for nodes within an MCTS tree
  """
  def __init__(self, gamestate: GameNode, action=""):
    self.N = 0
    self.Q = 0.0
    self.gamestate = gamestate
    self.action = action
    self.expanded = False
    self.children = []

  def expand(self) -> None:
    "fills the children list in the node"
    assert not self.is_expanded(), "calling expand on expanded node is bad"
    # if self.is_expanded:
      # return
    for move in self.gamestate.legal_moves():
      self.children.append(Node(self.gamestate.execute_move(move), move))
    self.expanded = True

  def is_expanded(self) -> bool:
    return self.expanded

  def __hash__(self) -> int:
    return hash(self.gamestate)+hash(self.action)

