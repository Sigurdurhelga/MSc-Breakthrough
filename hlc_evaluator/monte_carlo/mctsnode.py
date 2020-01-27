from abc import ABC, abstractmethod
class Node(ABC):
  """
    Generic class for states in a game to do MCTS on
  """

  @abstractmethod
  def get_children(self) -> set:
    "Returns all children for a state (after applying an action)"
    return set()

  @abstractmethod
  def find_random_child(self) -> object:
    "Returns a random child for this state"
    return None

  @abstractmethod
  def is_terminal(self) -> bool:
    "Returns true if this state is a terminal state"
    return True

  @abstractmethod
  def reward(self) -> int:
    """returns the reward for the current node (should be a terminal node
    otherwise methods other than mcts would be better)"""
    return 0

  @abstractmethod
  def __str__(self) -> str:
    "A node should have a string representation (it's used in error messages)"
    return ""

  @abstractmethod
  def __hash__(self) -> int:
    "a node has to be hashable to do efficient lookup"
    return 1

  @abstractmethod
  def __eq__(self, other) -> bool:
    "We need comparison of nodes"
    return True