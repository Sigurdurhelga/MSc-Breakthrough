import torch
import torch.nn as nn
from collections import defaultdict


class ConvBlock(nn.Module):
  def __init__(self, game_width, game_height, conv_filters=128):
    super(ConvBlock, self).__init__()
    self.conv_filters = conv_filters
    self.game_width = game_width
    self.game_height = game_height

    self.conv1      = nn.Conv2d(in_channels=7, out_channels=self.conv_filters, kernel_size=3, stride=1, padding=1, bias=False)
    self.batchnorm1 = nn.BatchNorm2d(self.conv_filters)
    self.relu1      = nn.ReLU()

  def forward(self,in_val):
    in_val = in_val.view(-1,7,self.game_width,self.game_height)
    output = self.conv1(in_val)
    output = self.batchnorm1(output)
    return self.relu1(output)

class ResBlock(nn.Module):
  def __init__(self, conv_filters=128, layer_name=""):
    super(ResBlock, self).__init__()
    self.conv_filters = conv_filters
    self.layer_name = layer_name

    self.conv1 = nn.Conv2d(
      in_channels=self.conv_filters,
      out_channels=self.conv_filters,
      kernel_size=3,
      stride=1,
      padding=1,
      bias=False
    )
    self.batchnorm1 = nn.BatchNorm2d(self.conv_filters)
    self.relu1      = nn.ReLU()
    self.conv2 = nn.Conv2d(
      in_channels=self.conv_filters,
      out_channels=self.conv_filters,
      kernel_size=3,
      stride=1,
      padding=1,
      bias=False
    )
    self.batchnorm2 = nn.BatchNorm2d(self.conv_filters)
    self.relu2      = nn.ReLU()

  def forward(self, in_val):
    residual = in_val

    output = self.conv1(in_val)
    output = self.batchnorm1(output)
    output = self.relu1(output)

    output = self.conv2(in_val)
    output = self.batchnorm2(output)
    output += residual
    output = self.relu2(output)

    return output

class OutBlock(nn.Module):
  def __init__(self, game_width, game_height,game_move_amount, conv_filters=128):
    super(OutBlock, self).__init__()
    self.conv_filters = conv_filters
    self.game_width = game_width
    self.game_height = game_height
    self.game_move_amount = game_move_amount

    # Value head
    self.conv_v = nn.Conv2d(
      in_channels=self.conv_filters,
      out_channels=3,
      kernel_size=1
    )
    self.batchnorm_v = nn.BatchNorm2d(3)
    self.relu_v_1 = nn.ReLU()
    self.full_conn_1_v = nn.Linear(in_features=3*self.game_height*self.game_width, out_features=32)
    self.relu_v_2 = nn.ReLU()
    self.full_conn_2_v = nn.Linear(in_features=32, out_features=1)
    self.tahn_v = nn.Tanh()

    # Policy head
    self.conv_p = nn.Conv2d(
      in_channels=self.conv_filters,
      out_channels=32,
      kernel_size=1
    )
    self.batchnorm_p = nn.BatchNorm2d(32)
    self.relu_p_1 = nn.ReLU()
    self.log_softmax = nn.LogSoftmax(dim=1)
    # Last policy head has out_features = total amount of possible moves in game
    self.full_conn_p = nn.Linear(in_features=(self.game_height * self.game_width * 32), out_features=game_move_amount)

  def forward(self, in_val):
    value = self.conv_v(in_val)
    value = self.batchnorm_v(value)
    value = self.relu_v_1(value)
    value = value.view(-1, 3*self.game_height*self.game_width)
    value = self.full_conn_1_v(value)
    value = self.relu_v_2(value)
    value = self.full_conn_2_v(value)
    value = self.tahn_v(value)

    policy = self.conv_p(in_val)
    policy = self.batchnorm_p(policy)
    policy = self.relu_p_1(policy)

    policy = policy.view(-1, self.game_height*self.game_width*32)
    policy = self.full_conn_p(policy)
    policy = self.log_softmax(policy)

    return policy, value

class BreakThroughAlphaZero(nn.Module):
  def __init__(self,game_width, game_height, game_move_amount, conv_filters=128):
    super(BreakThroughAlphaZero, self).__init__()
    self.convBlock = ConvBlock(game_width, game_height, conv_filters)
    self.residualBlocks = []
    for i in range(2):
      setattr(self, f"res_{i}", ResBlock(conv_filters, "res_layer_{}".format(i)))

    self.outBlock = OutBlock(game_width, game_height, game_move_amount, conv_filters)

  def forward(self, in_val):
    return self.forward_1(self.forward_0(in_val))


  def forward_0(self, in_val):
    output = self.convBlock(in_val)
    for i in range(2):
      output = getattr(self, f"res_{i}")(output)
    return output

  def forward_1(self, in_val):
    output = self.outBlock(in_val)
    return output


class AlphaLoss(nn.Module):
  def __init__(self):
    super(AlphaLoss, self).__init__()

  def forward(self, y_value, x_value, y_policy, x_policy):
    # torch.set_printoptions(profile="full")

    # print("y_value",y_value.view(-1))
    # print("x_value",x_value)
    # print("y_policy",y_policy)
    # print("x_policy",x_policy)
    value_error = torch.mean((x_value - y_value.view(-1)) ** 2) # squared error
    # policy_error = -torch.sum(x_policy * y_policy)/y_policy.size()[0]
    policy_error = -torch.sum(x_policy * y_policy)/y_policy.size()[0]

    # print("value error:",value_error)
    # print("policy error:",policy_error)

    # print("sum val:",torch.sum(value_error))
    # exit()

    # policy_error = torch.sum(y_policy * y_policy)/ y_policy.size()[0]
    total_error = value_error + policy_error
    return total_error