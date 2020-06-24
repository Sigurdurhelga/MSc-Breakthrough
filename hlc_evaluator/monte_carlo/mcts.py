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
  def __init__(self):
    self.Qs = {}
    self.Ps = defaultdict(tuple)
    self.Ns = {}
    self.visited = set()
    self.temp = 1

  def get_best_child(self,state):
    assert state in self.visited, "cant get best child of unknown"

    children = state.children

    best_score = -float("inf")
    best_child = None

    for i,child in enumerate(children):
      if not child:
        continue
      if self.Qs[state][i] > best_score:
        best_child = child
        best_score = self.Qs[state][i]
    return best_child

  def rollout(self, state):

    if state.gamestate.is_terminal():
      winner = state.gamestate.reward()
      if winner == 1:
        return -1 if state.gamestate.player == -1 else 1
      else:
        return 1 if state.gamestate.player == -1 else -1

    if state not in self.visited:
      self.visited.add(state)
      self.Qs[state] = [0] * state.gamestate.get_move_amount()
      self.Ns[state] = [0] * state.gamestate.get_move_amount()
      winner = self.simulate(state)
      if winner == 1:
        return -1 if state.gamestate.player == -1 else 1
      else:
        return 1 if state.gamestate.player == -1 else -1

    best_uct = -float('inf')
    best_child = None

    if not state.is_expanded():
      state.expand()

    children = state.children

    parent_n = sum(self.Ns[state])

    uct_array = []

    for i,child in enumerate(children):
      if not child:
        uct_array.append(-float("inf"))
        continue
      uct = self.Qs[state][i] + self.temp * (np.sqrt(parent_n) / (1+self.Ns[state][i]))
      uct_array.append(uct)

    qu_array = np.array(qu_array)
    qumax = np.max(qu_array)

    best_child = np.random.choice(np.argwhere(qu_array == qumax).flatten())
    
    next_board = state.gamestate.execute_move(children[best_child].action)
    
    next_state = Node(next_board,children[best_child].action)

    val = self.rollout(next_state)

    self.Qs[state][best_child] = (self.Ns[state][best_child] * self.Qs[state][best_child] + val) / (1+self.Ns[state][best_child])
    self.Ns[state][best_child] += 1
    return -val


  def nn_rollout(self,state,neural_network):


    if state.gamestate.is_terminal():
      winner = state.gamestate.reward()
      if winner == 1:
        return -1 if state.gamestate.player == -1 else 1
      else:
        return 1 if state.gamestate.player == -1 else -1

    if state not in self.visited:
      self.visited.add(state)
      self.Qs[state] = [0] * state.gamestate.get_move_amount()
      self.Ns[state] = [0] * state.gamestate.get_move_amount()
      policy, value = neural_network.safe_predict(state.gamestate)
      self.Ps[state] = (policy,value)
      # if state.gamestate.player == 1:
        # value = -value
      return -value
    else:
      policy,value = self.Ps[state]

    policy = policy.detach().cpu().numpy().reshape(-1)
    value = value.item()
    # as nn isn't negamaxed we should invert value if we're playing black
    # if state.gamestate.player == 1:
      # value = -value

    if not state.is_expanded():
      state.expand()

    children = state.children

    # Alphazero Selection process
    # Argmax(Q + U)
    # Q = W / N
    # U = temp * P(Action) * (sum(N_ALLactions) / N_action)

    best_qu = -float("inf")
    best_child = None

    parent_n = np.sqrt(sum(self.Ns[state]))
    
    qu_array = []

    for i,child in enumerate(children):
      if not child:
        qu_array.append(-float("inf"))
        continue
      qu_array.append(self.Qs[state][i] + self.temp * policy[i] * (parent_n / (1 + self.Ns[state][i])))

    qu_array = np.array(qu_array)
    qumax = np.max(qu_array)

    best_child = np.random.choice(np.argwhere(qu_array == qumax).flatten())

    next_board = state.gamestate.execute_move(children[best_child].action)
    
    next_state = Node(next_board,children[best_child].action)

    val = self.nn_rollout(next_state,neural_network)

    self.Qs[state][best_child] = ((self.Ns[state][best_child] * self.Qs[state][best_child]) + val) / (1+self.Ns[state][best_child])
    self.Ns[state][best_child] += 1
    return -val

  def get_policy(self,state,simulations,nnet, temp=1):

    for _ in range(simulations):
      self.nn_rollout(state, nnet)

    children = state.children

    policy = [self.Ns[state][i] if child else 0 for i,child in enumerate(children)]

    if temp == 0:
      # CONSIDER LOOKING AT ALL MAX MOVES AND PICKING RANDOM
      best_child = np.argmax(policy)
      policy[best_child] = 1.0

    else:
      policy = [x ** (1.0 / temp) for x in policy]
      psum = sum(policy)
      if psum == 0:
        psum = 1

      policy = [x/psum for x in policy]
    return np.array(policy)

  def simulate(self,node):
    """
      simulate(node)
        - does random moves from `node` until a terminal state is reached, returns the reward for that state
    """
    curr_node = deepcopy(node.gamestate)
    while not curr_node.is_terminal():
      moves = curr_node.legal_moves
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
