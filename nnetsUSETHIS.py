import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, 1) #1 channel input, 64 channel output, 3 filter size, 1 stride
        self.conv2 = nn.Conv2d(64, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        self.conv4 = nn.Conv2d(64, 64, 3, 1)
        self.conv5 = nn.Conv2d(64, 64, 3, 1)
        self.fc1 = nn.Linear(256, 512) #256=output of the cnn, 256=hidden layer
        self.fc2 = nn.Linear(512, 10) #256 hidden layers, 10 classes

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.leaky_relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.leaky_relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.leaky_relu(self.conv4(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.leaky_relu(self.conv5(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(x.shape[0], -1)
        x = F.leaky_relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1) #probability