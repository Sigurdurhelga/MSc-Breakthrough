from collections import defaultdict
from mctsnode import Node
import math

class MCTS:
  """
    'Monte Carlo Tree Search

    do random moves to average the score for a given node, this is good when it's
    hard to actually evaluate a position on a board given just the position,
    then we just simulate a bunch of moves from it and average the results

    variables of MCTS:
      Q total reward from nodes
      N total visit count for nodes
      children array of children nodes for a node
      exploration weight constant

    notes:
      if a node is in the children dict it is expanded

  """
  def __init__(self, negamaxing=False, exploration_weight=1):
    self.Q = defaultdict(int)
    self.N = defaultdict(int)
    self.children = dict()
    self.exploration_weight = exploration_weight
    self.negamaxing = negamaxing

  def mcts_score(self, node) -> float:
    if self.N[node] == 0:
      # unknown nodes are not considered
      return float("-inf")
    "Average Score - is the sum of its rewards divided by the amount of visits"
    return self.Q[node] / self.N[node]

  def choose_child(self, node: Node) -> Node:
    "Choose the best child of a node"
    assert(not node.is_terminal(), "Calling choose child on a terminal node, is not allowed")

    if node not in self.children:
      return node.find_random_child()

    return max(self.children[node], key=self.mcts_score)

  def do_rollout(self, node: Node) -> None:
    """
      Rollout of nodes. We expand a node and do random moves to estimate the
      actual undelying reward from doing the move
    """
    path = self.select(node)
    leaf = path[-1]
    self.expand(leaf)
    reward = self.mcts_simulate(leaf)
    self.backpropagate(path, reward)

  def select(self, curr_node: Node) -> list:
    """
      function should not be called directly
      it's called during rollout.

      Function finds an unexpanded descendant of *node*
    """
    path = []
    while True:
      path.append(curr_node)
      if curr_node not in self.children or curr_node.is_terminal():
        return path
      unexplored = self.children[curr_node] - self.children.keys()
      if unexplored:
        end = unexplored.pop()
        path.append(end)
        return path
      curr_node = self.uct_select(curr_node) # randomly select a child
    raise RuntimeError("Select didn't find a path")

  def mcts_simulate(self, node: Node) -> float:
    while True:
      if node.is_terminal():
        return node.reward() if not self.negamaxing else -node.reward()
      node = node.find_random_child()

  def expand(self, node: Node) -> None:
    if node in self.children:
      return
    self.children[node] = node.get_children()

  def backpropagate(self, path, reward) -> None:
    for node in reversed(path):
      self.N[node] += 1
      self.Q[node] += reward
      # reward = 0 - ( reward ) # invert from -1/1 every iteration

  def uct(self, node: Node, parent_N: int) -> float:
    "returns the upper confidence bound"
    assert(node in self.children, "UCT called on node that hadn't been expanded")
    parent_log_N = math.log(parent_N)
    return (self.Q[node] / self.N[node]) + (self.exploration_weight * math.sqrt(parent_log_N / self.N[node]))

  def uct_select(self, node: Node) -> Node:
    "select a child of node, randomly w.r.t exploration/exploitation"
    node_N = self.N[node]
    children_uct = [(n,self.uct(n, node_N)) for n in self.children[node]]

    return max(children_uct, key=lambda x: x[1])[0]
