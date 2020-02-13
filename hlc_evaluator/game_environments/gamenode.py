from __future__ import annotations
from abc import ABC, abstractmethod
from copy import copy

class GameNode(ABC):
  @abstractmethod
  def is_terminal(self) -> bool:
    pass

  @abstractmethod
  def reward(self) -> int:
    pass

  @abstractmethod
  def legal_moves(self) -> list:
    pass

  @abstractmethod
  def execute_move(self, move) -> GameNode:
    pass

  @abstractmethod
  def __copy__(self):
    pass

  @abstractmethod
  def __hash__(self) -> int:
    pass

  @abstractmethod
  def __eq__(self,other) -> bool:
    pass