from neural_networks.neuralnetworkbase import NNBase
from neural_networks.breakthrough.neuralnetwork import BreakThroughAlphaZero, AlphaLoss
import numpy as np

from collections import namedtuple
import os
import torch
import torch.nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils import clip_grad_norm_

config_neuralnet = namedtuple("nnconfig", "lr epochs batch_size conv_filters cuda grad_steps")

config = config_neuralnet(0.01,
                          5,
                          128,
                          128,
                          torch.cuda.is_available(),
                          10)

class BoardData(Dataset):
  def __init__(self, dataset):
    # super(BoardData, self).__init__()
    self.X = dataset

  def __len__(self):
    return len(self.X)

  def __getitem__(self, idx):
    xboard, xpolicy, xvalue = self.X[idx]
    xboard = torch.FloatTensor(xboard.transpose(2,0,1))
    xpolicy = torch.FloatTensor(np.array(xpolicy))
    xvalue = torch.FloatTensor(np.array(xvalue).astype(np.float64))

    return xboard,xpolicy,xvalue

class BreakthroughNN(NNBase):
  def __init__(self, game_width, game_height, game_move_amount):
    super(BreakthroughNN, self).__init__()
    self.neural_network = BreakThroughAlphaZero(game_width, game_height, game_move_amount, config.conv_filters)
    if config.cuda:
      self.neural_network.cuda()
    self.neural_network.eval()
    self.optimizer = optim.Adam(self.neural_network.parameters())
    self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[100,200,400,800], gamma=0.777)


  def predict(self, example):
    if type(example) != np.ndarray:
      example = example.encode_state()
    example = torch.FloatTensor(example.transpose(2,0,1))
    if config.cuda:
      example = example.cuda()
    return self.neural_network(example)

  def safe_predict(self, example):
    if type(example) != np.ndarray:
      example = example.encode_state()
    example = torch.FloatTensor(example.transpose(2,0,1))
    if config.cuda:
      example = example.cuda()
    output = None
    with torch.no_grad():
      output = self.neural_network(example)
    return output

  def train(self, dataset):
    criterion = AlphaLoss()
    dataset = BoardData(dataset)
    train_data = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, pin_memory=False)
    losses = []
    for epoch in range(config.epochs):
      total_loss = 0
      for batch in train_data:
        xboard, xpolicy, xvalue = batch

        if config.cuda:
          xboard = xboard.cuda()
          xpolicy =xpolicy.cuda()
          xvalue = xvalue.cuda()


        ypolicy, yvalue = self.neural_network(xboard)
        if config.cuda:
          ypolicy = ypolicy.cuda()
          yvalue = yvalue.cuda()
        loss = criterion(yvalue, xvalue, ypolicy, xpolicy)
        if config.cuda:
          loss.cuda()

        loss.backward()
        clip_grad_norm_(self.neural_network.parameters(), 1)

        self.optimizer.step()
        self.optimizer.zero_grad()

        total_loss += loss.item()

      self.scheduler.step()
      losses.append(total_loss)
    # print(f"[breakthrough_nn.py train()] total losses for training were {losses}")

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
    if config.cuda:
      self.neural_network.cuda()

