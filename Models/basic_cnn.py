import torch.nn as nn
import torch.nn.functional as F
import os

classes = os.listdir('Data/intel/seg_test')


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 8, stride=2)
        self.pool = nn.MaxPool2d(3, 3)
        self.conv2 = nn.Conv2d(6, 16, 8, stride=2)
        self.fc1 = nn.Linear(144, 120)
        self.fc2 = []
        for i in range(3):
            self.fc2.append(nn.Linear(120, 120))
        self.fc3 = nn.Linear(120, 84)
        self.fc4 = nn.Linear(84, len(classes))

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 144)
        x = F.relu(self.fc1(x))
        for i in range(3):
            x = F.tanh(self.fc2[i](x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


net = Net()
