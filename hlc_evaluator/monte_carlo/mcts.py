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
  """
  def __init__(self, root_node:Node, neural_network=None):
    self.root = root_node
    self.initial_root = root_node
    self.Qs = defaultdict(float)
    self.Ps = defaultdict(float)
    self.Ns = defaultdict(int)
    self.temp = 1
    self.neural_network = neural_network

  def move_to_best_child(self):
      """
        move_to_best_child()
          - moves the root in the tree to the best child of the current root
      """
      assert self.root.is_expanded(), "get best child on unexpanded node, is bad"

      children = self.root.children
      child_scores = [self.Qs[child] for child in children]
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

        curr_node = curr_node.children[np.argmax(child_scores)]

      if uct_terminal:
        reward = path[-1].gamestate.reward()
        self.backpropagate(reward, path)

      else:
        end = path[-1]
        #print("did rollout with path",path)
        end.expand()
        reward = self.simulate(end)
        self.backpropagate(reward, path)

  def nn_rollout(self, rollout_amount=10):
    """
      rollout(rollout_amount)
        - does rollout_amount many monte carlo simulations from the root
    """
    assert self.neural_network, "Can't call get_training_data() on a MCTS without a neural network"

    # This is not general, should be fixed for other games (maybe a method of gamenodes for node -> policy idx)
    width = self.root.gamestate.cols
    depth = 6

    for _ in range(rollout_amount):
      curr_node = self.root
      path = []

      ## SELECTION PROCESS
      while True:
        path.append(curr_node)
        self.Ns[curr_node] += 1

        policy, value = self.neural_network.predict(curr_node.gamestate)
        policy = policy.detach().cpu().numpy().reshape(-1)
        value = value.item()
        self.backpropagate(value, path)

        if not curr_node.is_expanded():
          curr_node.expand()
          break
        if curr_node.gamestate.is_terminal():
          break

        children = curr_node.children

        action_idxs = []
        # BREAKTRHOUGH SPECIFIC, GOT TO FIX FOR GENERALITY (see above non-generality comment)
        # Flatten the index as policy is a 1d array representing 3d array (height * width * action_size)
        for idx,child in enumerate(children):
          x1,y1,x2,y2 = child.action
          direction = 0
          direction += 0 if x1 < x2 else 3
          direction += 0 if y1 < y2 else 1 if y1 == y2 else 2
          action_idxs.append((x2 + (width * (y2 + (depth * direction))),idx))

        mask_idxs = [i for i in range(len(policy)) if i not in [x[0] for x in action_idxs]]
        policy[mask_idxs] = 0

        # Alphazero Selection process
        # Argmax(Q + U)
        # Q = W / N
        # U = temp * P(Action) * (sum(N_ALLactions) / N_action)
        all_action_sum = 0
        node_qu = []
        all_action_sum = sum(self.Qs[child]/(self.Ns[child]+1) for child in children)
        for pidx, idx in action_idxs:
          child = children[idx]
          child_q = self.Qs[child] / (self.Ns[child]+1)
          child_u = self.temp * policy[pidx] * (all_action_sum / (self.Ns[child]+1))
          node_qu.append(child_q + child_u)

        curr_node = curr_node.children[np.argmax(node_qu)]

  def get_policy(self, temp=1):
    children = self.root.children
    if temp == 0: # select only best
      pi = [0] * len(children)
      visits = [self.Ns[child] for child in children]
      pi[np.argmax(visits)] = 1.0

    else:
      total_child_visits = sum(self.Ns[child] for child in children) ** (1/temp)
      pi = [self.Ns[child] ** (1/temp)/total_child_visits for child in children]
    return pi

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
      reward = -reward
