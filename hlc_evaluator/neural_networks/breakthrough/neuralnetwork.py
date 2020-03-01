import torch
import torch.nn as nn

class ConvBlock(nn.Module):
  def __init__(self, game_width, game_height, conv_filters=128):
    super(ConvBlock, self).__init__()
    self.conv_filters = conv_filters
    self.game_width = game_width
    self.game_height = game_height

    self.conv1      = nn.Conv2d(in_channels=3, out_channels=self.conv_filters, kernel_size=3, stride=1, padding=1, bias=False)
    self.batchnorm1 = nn.BatchNorm2d(self.conv_filters)
    self.relu1      = nn.ReLU()

  def forward(self,in_val):
    in_val = in_val.view(-1,3,self.game_width,self.game_height)
    output = self.conv1(in_val)
    output = self.batchnorm1(output)
    return self.relu1(output)

class ResBlock(nn.Module):
  def __init__(self, conv_filters=128):
    super(ResBlock, self).__init__()
    self.conv_filters = conv_filters

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
    self.logsoftmax = nn.LogSoftmax(dim=1)
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
    policy = self.softmax(policy)

    return policy, value

class BreakThroughAlphaZero(nn.Module):
  def __init__(self,game_width, game_height, game_move_amount, conv_filters=128):
    super(BreakThroughAlphaZero, self).__init__()
    self.convBlock = ConvBlock(game_width, game_height, conv_filters)
    self.residualBlocks = []
    for i in range(5):
      setattr(self, f"res_{i}", ResBlock(conv_filters))

    self.outBlock = OutBlock(game_width, game_height, game_move_amount, conv_filters)

  def forward(self, in_val):
    output = self.convBlock(in_val)
    for i in range(5):
      output = getattr(self, f"res_{i}")(output)
    output = self.outBlock(output)

    return output

class AlphaLoss(nn.Module):
  def __init__(self):
    super(AlphaLoss, self).__init__()

  def forward(self, y_value, x_value, y_policy, x_policy):
    value_error = (x_value - y_value) ** 2 # squared error
    policy_error = torch.sum((-x_policy * (1e-8 + y_policy.float()).float().log()), 1)
    total_error = (value_error.view(-1).float() + policy_error).mean()
    return total_error