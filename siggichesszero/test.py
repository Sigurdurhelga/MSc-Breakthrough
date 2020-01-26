import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets

class ConvBlock(nn.Module):
  def __init__(self):
    super(ConvBlock, self).__init__()
    self.conv1      = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
    self.batchnorm1 = nn.BatchNorm2d(128)
    self.relu1      = nn.ReLU()

  def forward(self,input):
    output = self.conv1(input)
    output = self.batchnorm1(output)
    return self.relu1(output)

class ResBlock(nn.Module):
  def __init__(self):
    super(ResBlock, self).__init__()
    self.conv1 = nn.Conv2d(
      in_channels=128,
      out_channels=128,
      kernel_size=3,
      stride=1,
      bias=False
    )
    self.batchnorm1 = nn.BatchNorm2d(128)
    self.relu1      = nn.ReLU()
    self.conv2 = nn.Conv2d(
      in_channels=128,
      out_channels=128,
      kernel_size=3,
      stride=1,
      bias=False
    )
    self.batchnorm2 = nn.BatchNorm2d(128)
    self.relu2      = nn.ReLU()

  def forward(self, input):
    residual = input
    output = self.conv1(input)
    output = self.batchnorm1(output)
    output = self.relu1(output)
    output = self.conv2(input)
    output = self.batchnorm2(output)
    # add the residual before reluing
    output += residual
    output = self.relu2(output)
    return output

class OutBlock(nn.Module):
  def __init__(self):
    super(OutBlock, self).__init__()
    # Value head
    self.conv_v = nn.Conv2d(
      in_channels=128,
      out_channels=3,
      kernel_size=1
    )
    self.batchnorm_v = nn.BatchNorm2d(3)
    self.relu_v_1 = nn.ReLU()
    self.full_conn_1_v = nn.Linear(128, 32)
    self.relu_v_2 = nn.ReLU()
    self.full_conn_2_v = nn.Linear(32,1)
    self.tahn_v = nn.Tanh()

    # Policy head
    self.conv_p = nn.Conv2d(
      in_channels=128,
      out_channels=32,
      kernel_size=1
    )
    self.batchnorm_p = nn.BatchNorm2d(32)
    self.relu_p_1 = nn.ReLU()
    self.logsoftmax = nn.LogSoftmax(dim=1)
    self.full_conn_p = nn.Linear(6*7*32, 7)

  def forward(self, input):
    value = self.conv_v(input)
    value = self.batchnorm_v(value)
    value = self.relu_v(value)
    print("in forward before going into fully connected stuff maybe gotta do view stuff",value)
    value = self.full_conn_1_v(value)
    value = self.relu_v_2(value)
    value = self.full_conn_2_v(value)
    value = self.tahn_v(value)

    policy = self.conv_p(input)
    policy = self.batchnorm_p(policy)
    policy = self.relu_p_1(policy)
    print("in forward SECOND before going into fully connected stuff maybe gotta do view stuff",value)
    policy = self.full_conn_p(policy)
    policy = self.logsoftmax(policy).exp()

    return policy, value

class AlphaZero(nn.Module):
  def __init__(self):
    super(AlphaZero, self).__init__()
    self.convBlock = ConvBlock()
    self.residualBlocks = []
    for i in range(19):
      self.residualBlocks.append(ResBlock())
    self.outBlock = OutBlock()

  def forward(self, input):
    output = self.convBlock(input)
    for resBlock in self.residualBlocks:
      output = resBlock(output)
    output = self.outBlock(output)

    return output

class AlphaLoss(nn.Module):
  def __init__(self):
    super(AlphaLoss, self).__init__()

  def forward(self, y_value, value, y_policy, policy):
    value_error = (value - y_value) ** 2 # squared error

    policy_error = torch.sum((-policy * (1e-8 + y_policy.float()).float().log()))

    total_error = (value_error.view(-1).float() + policy_error).mean()

    return total_error





# initializing models
class CNNModel(nn.Module):
    def __init__(self):
        """
          pytorch init class for setting up the whole model,
          we call the super class' init function as it sets
          this up as a layer in the network. Check forward()
          to get an understanding about how this works
        """
        super(CNNModel, self).__init__()
        # Convolution 1
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        # Convolution 2
        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        # Fully connected 1 (readout)
        self.fc1 = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        """
          basic functionality of this network, forward is the
          function that is called with the input. We just pass
          it through all our layers here.
        """
        # Convolution 1
        out = self.cnn1(x)
        out = self.relu1(out)

        # Max pool 1
        out = self.maxpool1(out)

        # Convolution 2
        out = self.cnn2(out)
        out = self.relu2(out)

        # Max pool 2
        out = self.maxpool2(out)

        # Resize
        # Original size: (100, 32, 7, 7)
        # out.size(0): 100
        # New out size: (100, 32*7*7)
        out = out.view(out.size(0), -1)

        # Linear function (readout)
        out = self.fc1(out)

        return out

# initializing dataset
print("Initializing dataset")
print("====================")
train_dataset = dsets.MNIST(root='./data',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)

test_dataset = dsets.MNIST(root='./data',
                          train=False,
                          transform=transforms.ToTensor())

print("size of training data {}".format(train_dataset.data.size()))
print("size of testing data {}".format(test_dataset.data.size()))
print("====================")

# initializing dataloader
print("Initializing dataloader")
print("====================")
batch_size = 100
n_iters = 3000
num_epochs = n_iters / (len(train_dataset) / batch_size)
num_epochs = int(num_epochs)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                          batch_size=batch_size,
                                          shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)
print("====================")

# setting up the neural network
print("Setting up model")
print("====================")
model = AlphaZero()

model_path = ''

if model_path:
  model.load_state_dict(torch.load(model_path))

else:
  criterion = AlphaLoss()
  print("====================")

  learning_rate = 0.01
  optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

print("Examining model")
print("====================")
# examination of the neural network post setup
print(model.parameters())
print(len(list(model.parameters())))

for i in list(model.parameters()):
  print(i.size())
print("====================")

iter = 0
for epoch in range(num_epochs):
    print("Starting EPOCH {} | {}".format(epoch, num_epochs))
    for i, (images, labels) in enumerate(train_loader):
        print("image {} out of {}".format(i,train_loader.batch_size))
        print("trainloader image {}".format(labels))
        # Load images
        images = images.requires_grad_()

        # Clear gradients w.r.t. parameters
        optimizer.zero_grad()

        # Forward pass to get output/logits
        outputs = model(images)

        # Calculate Loss: softmax --> cross entropy loss
        loss = criterion(outputs, labels)

        # Getting gradients w.r.t. parameters
        loss.backward()

        # Updating parameters
        optimizer.step()

        iter += 1

        if iter % 500 == 0:
            print("Testing on testdata")
            # Calculate Accuracy
            correct = 0
            total = 0
            # Iterate through test dataset
            for images, labels in test_loader:
                # Load images
                images = images.requires_grad_()

                # Forward pass only to get logits/output
                outputs = model(images)

                # Get predictions from the maximum value
                _, predicted = torch.max(outputs.data, 1)

                # Total number of labels
                total += labels.size(0)

                # Total correct predictions
                correct += (predicted == labels).sum()

            accuracy = 100 * correct / total

            # Print Loss
            print('Iteration: {}. Loss: {}. Accuracy: {}'.format(iter, loss.item(), accuracy))

from datetime import datetime
torch.save(model.state_dict(), './{}_{}_{}_{}_model.h5'.format(datetime.month, datetime.day, datetime.hour, datetime.minute))