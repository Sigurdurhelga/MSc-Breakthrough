from __future__ import annotations
from abc import ABC, abstractmethod

class NNBase(ABC):
  @abstractmethod
  def __init__(self):
    """
      Initialize a neural network for a game
    """
    pass

  @abstractmethod
  def predict(self, example):
    """
      Given an example return P,v (policy and value) for the example (boardstate)
    """
    pass

  @abstractmethod
  def train(self, dataset):
    """
      Takes a dataset of (boardstate, policy, value) and trains the neural network
      using the data
    """
    pass

  @abstractmethod
  def savemodel(self, folder:str, filename:str) -> None:
    """
      Saves the model as .h5 file in the folder with the filename+.h5
    """
    pass

  @abstractmethod
  def loadmodel(self, folder:str, filename:str) -> NNBase:
    """
      Loads the model from a .h5 file in the folder+filename
    """
    pass