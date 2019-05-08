import torchvision
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import transforms, datasets

import os
import matplotlib.pyplot as plt
import numpy as np


# class ImageFolderWithPaths(datasets.ImageFolder):
#     def __getitem__(self, index):
#         original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
#         path = self.imgs[index][0]
#         tuple_with_path = (original_tuple + (path,))
#         return tuple_with_path

if __name__ == '__main__':
    torch.manual_seed(123)
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = datasets.ImageFolder(root='Data/intel/seg_train', transform=transform)
# trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
#                                        download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

    testset = datasets.ImageFolder(root='Data/intel/seg_test', transform=transform)
# testset = torchvision.datasets.CIFAR10(root='./data', train=False,
#                                       download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=4, shuffle=True, num_workers=2)

    classes = os.listdir('Data/intel/seg_test')

# net = torchvision.models.resnet18(pretrained=True)
# for param in net.parameters():
#     param.requires_grad = False
#
# num_ftrs = net.fc.in_features
# net.fc = nn.Sequential(nn.Linear(num_ftrs, num_ftrs),
#                        nn.Tanh(), nn.Linear(num_ftrs, num_ftrs),
#                        nn.Tanh(), nn.Linear(num_ftrs, num_ftrs),
#                        nn.Tanh(), nn.Linear(num_ftrs, num_ftrs),
#                        nn.Tanh(), nn.Linear(num_ftrs, num_ftrs),
#                        nn.Tanh(), nn.Linear(num_ftrs, len(classes))
#                        )


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
                x = torch.tanh(self.fc2[i](x))
            x = F.relu(self.fc3(x))
            x = self.fc4(x)
            return x


    net = Net()
    net.load_state_dict(torch.load("transfer-4-tanh.pt"))
    net.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the test images: %d %%' % (
            100 * correct / total))

# What classes perform well?
    class_correct = list(0. for i in range(len(classes)))
    class_total = list(0. for i in range(len(classes)))
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            if len(labels) != 1:
                c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(len(classes)):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))
