from monte_carlo.mctsnode import Node
from collections import defaultdict, namedtuple
from copy import deepcopy
import numpy as np
from random import choice

class MCTS():
  def __init__(self, root_node:Node, negamaxing=False):
    self.root = root_node
    self.Qs = defaultdict(float)
    self.Ps = defaultdict(float)
    self.Ns = defaultdict(int)
    # self.Usa = defaultdict(float)
    self.temp = 0.1
    self.negamaxing = negamaxing

  def move_to_best_child(self):
      assert self.root.is_expanded(), "get best child on unexpanded node, is bad"
      children = self.root.children
      child_scores = [self.Qs[child] for child in children]
      if self.negamaxing:
        best_move = np.argmin(child_scores)
      else:
        best_move = np.argmax(child_scores)
      self.root = children[best_move]

  def move_to_child(self, move):
    if not self.root.is_expanded():
      self.root.expand()
    children = self.root.children
    for child in children:
      if child.action == move:
        self.root = child
        return
    assert False, "Move to child didn't find move"

  def set_node(self, node: Node):
    self.root = node

  def rollout(self, rollout_amount=10):
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

  def simulate(self, node):
    curr_node = deepcopy(node.gamestate)
    while not curr_node.is_terminal():
      moves = curr_node.legal_moves()
      curr_node = curr_node.execute_move(choice(moves))
    return curr_node.reward()

  def backpropagate(self, reward, path):
    for state in reversed(path):
      self.Qs[state] += reward
