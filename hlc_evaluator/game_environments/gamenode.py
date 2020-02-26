from __future__ import annotations
from abc import ABC, abstractmethod
from copy import copy
import numpy as np

class GameNode(ABC):
  """
    Generic Abstract Class for a game environment
  """
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
  def encode_state(self) -> np.ndarray:
    pass

  @abstractmethod
  def get_move_amount(self) -> int:
    pass

  @abstractmethod
  def initial_state(self) -> GameNode:
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