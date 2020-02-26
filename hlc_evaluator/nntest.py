import torch
import torch.optim as optim
import numpy as np
from game_environments.breakthrough.breakthrough import BTBoard,config
from neural_networks.breakthrough.breakthrough_nn import BreakthroughNN
from timeit import timeit

a1 = np.array([
  [config.BLACK,config.BLACK],
  [config.BLACK,config.BLACK],
  [config.EMPTY,config.EMPTY],
  [config.EMPTY,config.EMPTY],
  [config.EMPTY,config.EMPTY],
  [config.WHITE,config.WHITE],
  [config.WHITE,config.WHITE],
])

b1 = BTBoard(a1, config.WHITE)
net = BreakthroughNN(b1.rows, b1.cols, b1.get_move_amount())
for i in range(100):
  start = timeit()
  p,v = net.predict(b1)
  total = timeit() - start
  print(total)

print(v.item())
print("=======")
print(p.detach().cpu().numpy().reshape(-1))
