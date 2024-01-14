import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class LeNet5(nn.Module):
    # network structure
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(46656, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc = nn.Linear(84, 10)

    def forward(self, x):
        """
        One forward pass through the network.

        Args:
            x: input
        """
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc(x)
        return x

    def num_flat_features(self, x):
        """
        Get the number of features in a batch of tensors `x`.
        """
        size = x.size()[1:]
        return np.prod(size)
