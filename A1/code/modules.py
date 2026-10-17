import torch.nn as nn
import torch.nn.functional as F




class DynamicMLP(nn.Module):
    def __init__(self, in_feat, out_feat, K, D):
        """
        Module with fixed width D at each hidden layer.
        With K=0, it is a simple Perceptron,
        with K>=1 it is an MLP.
        """
        if K <= 0:
            self.layers = nn.ModuleList([nn.Linear(in_feat, out_feat)])
        else:
            self.layers = nn.ModuleList([nn.Linear(in_feat, D)])
            for _ in range(K-1):
                self.layers.append(nn.Linear(D, D))
            self.layers.append(nn.Linear(D, out_feat))
        self.activation = nn.ReLU()

    def forward(self, x):
        return [self.activation(lay(x)) for lay in self.layers][-1]


class LinearReLU(nn.Module):
    def __init__(self, in_feat, out_feat, K, D):


class Conv2DPool(nn.Module):
    def __init__(self, in_feat, out_feat, K, D):
        super().__init()
        self.conv2drelupool_stack = nn.Sequential(
            nn.Conv2d(),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
        )

    def forward(self, x):
        return self.conv2drelupool_stack(x)

class DynamicCNN(nn.module):
    def __init__(self, in_feat, out_feat, K, D):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()