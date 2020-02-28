from neural_networks.neuralnetworkbase import NNBase
from neural_networks.breakthrough.neuralnetwork import BreakThroughAlphaZero, AlphaLoss
import numpy as np

from collections import namedtuple
import os
import torch
import torch.nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

config_neuralnet = namedtuple("nnconfig", "lr epochs batch_size conv_filters cuda grad_steps")

config = config_neuralnet(0.01,
                          5,
                          32,
                          128,
                          torch.cuda.is_available(),
                          10)



class BreakthroughNN(NNBase):
  def __init__(self, game_width, game_height, game_move_amount):
    super(BreakthroughNN, self).__init__()
    self.neural_network = BreakThroughAlphaZero(game_width, game_height, game_move_amount, config.conv_filters)
    self.neural_network.eval()
    self.optimizer = optim.Adam(self.neural_network.parameters())
    self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[100,200,400,800], gamma=0.777)


  def predict(self, example):
    if type(example) != np.ndarray:
      example = example.encode_state()
    # example = torch.from_numpy(example.transpose(2,0,1)).float()
    example = torch.FloatTensor(example.transpose(2,0,1))
    return self.neural_network(example)

  def train(self, dataset):
    batch_amount = 10
    batch_size = len(dataset) // batch_amount

    criterion = AlphaLoss()
    train_data = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=config.cuda)


    for epoch in range(config.epochs):
      total_loss = 0
      for batch_idx in range(batch_amount):
        for example in dataset[batch_size*batch_idx : batch_size * (batch_idx + 1)]:
          xboard, xpolicy, xvalue = example
          xboard = torch.FloatTensor(xboard.transpose(2,0,1))
          xpolicy = torch.FloatTensor(np.array(xpolicy))
          xvalue = torch.FloatTensor(np.array(xvalue).astype(np.float64))
          if config.cuda:
            xboard = xboard.cuda()
            xpolicy = xpolicy.cuda()
            xvalue = xvalue.cuda()

          ypolicy, yvalue = self.neural_network(xboard)
          loss = criterion(yvalue, xvalue, ypolicy, xpolicy)

          loss /= config.grad_steps
          loss.backward()

          clip_grad_norm_(self.neural_network.parameters(), 1)

          if epoch % config.grad_steps == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()

          total_loss += loss.item()

        self.scheduler.step()

  def savemodel(self, path, filename):
    filepath = os.path.join(path,filename)
    if not os.path.exists(path):
      os.mkdir(path)

    torch.save({
      'model_state_dict': self.neural_network.state_dict(),
      'optimizer_state_dict': self.optimizer.state_dict(),
      'scheduler_state_dict': self.scheduler.state_dict(),
    }, filepath)

  def loadmodel(self, path, filename):
    filepath = os.path.join(path,filename)
    if not os.path.exists(filepath):
      self.savemodel(path, filename)
      return
    map_location = None if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(filepath, map_location=map_location)
    self.neural_network.load_state_dict(checkpoint["model_state_dict"])
    self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

