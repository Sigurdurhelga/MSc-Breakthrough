from monte_carlo.mctsnode import Node
from collections import defaultdict, namedtuple
from copy import deepcopy
import numpy as np
from random import choice

class MCTS():
  """
    Class representing a montecarlo tree and to do monte carlo tree search

    Member variables:
      root - mctsnode.Node       - Node representing the current node the tree is examining (rollouts go from the root etc.)
      Qs   - dict[mctsnode.Node] - Stores the aggregate value of the node from backpropagation / simulation
      Ps   - dict[mctsnode.Node] - Stores the policy vector for a node
      Ns   - dict[mctsnode.Node] - Stores the visit count for a node
      temp - float               - float representing the temp (used in puct in alphazero see paper)
      negamaxing - bool          - Controls whether the node should select the maximum or the minimum on a node
  """
  def __init__(self, root_node:Node, negamaxing=False):
    self.root = root_node
    self.Qs = defaultdict(float)
    self.Ps = defaultdict(float)
    self.Ns = defaultdict(int)
    self.temp = 1
    self.negamaxing = negamaxing

  def move_to_best_child(self):
      """
        move_to_best_child()
          - moves the root in the tree to the best child of the current root
      """
      assert self.root.is_expanded(), "get best child on unexpanded node, is bad"
      children = self.root.children
      child_scores = [self.Qs[child] for child in children]
      if self.negamaxing:
        best_move = np.argmin(child_scores)
      else:
        best_move = np.argmax(child_scores)
      self.root = children[best_move]

  def move_to_child(self, move):
      """
        move_to_child(move)
          - moves the root in the tree to the child from doing `move` of the current root
      """
    if not self.root.is_expanded():
      self.root.expand()
    children = self.root.children
    for child in children:
      if child.action == move:
        self.root = child
        return
    assert False, "Move to child didn't find move"

  def set_node(self, node: Node):
    """
      set_node(node)
        - Force the root to a node
    """
    self.root = node

  def rollout(self, rollout_amount=10):
    """
      rollout(rollout_amount)
        - does rollout_amount many monte carlo simulations from the root
    """
    rollout_length_sum = 0
    for _ in range(rollout_amount):
      curr_node = self.root
      path = []
      uct_terminal = False
      while True:
        path.append(curr_node)
        self.Ns[curr_node] += 1
        if not curr_node.is_expanded():
          break
        if curr_node.gamestate.is_terminal():
          uct_terminal = True
          break

        children = curr_node.children
        # UCT FORMULA FOR ALL CHILDREN
        parent_nlog = np.log(self.Ns[curr_node])
        child_scores = [(self.Qs[child]/(self.Ns[child]+1)) + (self.temp*np.sqrt(parent_nlog/(1+self.Ns[child]))) for child in children]

        # Psa isn't filled as we don't have a neural network yet let's call it 1 for now
        #temp_PSA = 1

        curr_node = curr_node.children[np.argmax(child_scores)]
      rollout_length_sum += len(path)

      if uct_terminal:
        reward = path[-1].gamestate.reward()
        self.backpropagate(reward, path)

      else:
        end = path[-1]
        #print("did rollout with path",path)
        end.expand()
        reward = self.simulate(end)
        self.backpropagate(reward, path) # every iteration because of negamaxing
    # print("average rollout length",rollout_length_sum/rollout_amount)

  def simulate(self,node):
    """
      simulate(node)
        - does random moves from `node` until a terminal state is reached, returns the reward for that state
    """
    curr_node = deepcopy(self.root.gamestate)
    while not curr_node.is_terminal():
      moves = curr_node.legal_moves()
      curr_node = curr_node.execute_move(choice(moves))
    return curr_node.reward()

  def backpropagate(self, reward, path):
    """
      backpropagate(reward, path)
        - updates the Q values for all nodes on the path with the value gained from a simulation
    """
    for state in reversed(path):
      self.Qs[state] += reward
