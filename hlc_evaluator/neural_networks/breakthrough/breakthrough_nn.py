from neural_networks.neuralnetworkbase import NNBase
from neural_networks.breakthrough.neuralnetwork import BreakThroughAlphaZero, AlphaLoss
import numpy as np

from collections import namedtuple
import os
import torch
import torch.nn

config_neuralnet = namedtuple("nnconfig", "lr epochs batch_size conv_filters cuda")

config = config_neuralnet(0.01,
                          10,
                          32,
                          128,
                          torch.cuda.is_available())



class BreakthroughNN(NNBase):
  def __init__(self, game_width, game_height, game_move_amount):
    super(BreakthroughNN, self).__init__()
    self.neural_network = BreakThroughAlphaZero(game_width, game_height, game_move_amount, config.conv_filters)
    self.neural_network.eval()

    self.optimizer = AlphaLoss()

  def predict(self, example):
    if type(example) != np.ndarray:
      example = example.encode_state()
    example = torch.from_numpy(example.transpose(2,0,1)).float()
    return self.neural_network(example)

  def train(self, dataset):
    batch_size = len(dataset) // config.epochs
    batch_amount = len(dataset) // batch_size
    for epoch in range(config.epochs):
      for batch_idx in range(batch_amount):
        for example in dataset[batch_size*batch_idx : batch_size * (batch_idx + 1)]:
          xpolicy, xvalue, yresult = example
          pass
    pass

  def savemodel(self, path, filename):
    filepath = os.path.join(path,filename)
    if not os.path.exists(path):
      os.mkdir(path)

    torch.save({
      'model_state_dict': self.neural_network.state_dict(),
      'optimizer_state_dict': self.optimizer.state_dict()
    }, filepath)

  def loadmodel(self, path, filename):
    filepath = os.path.join(path,filename)
    if not os.path.exists(filepath):
      raise FileNotFoundError("Model file to load didn't exist")
    map_location = None if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(filepath, map_location=map_location)
    self.neural_network.load_state_dict(checkpoint["model_state_dict"])
    self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

