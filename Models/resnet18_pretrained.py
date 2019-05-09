import torch.nn as nn
import torchvision
import os


classes = os.listdir('Data/intel/seg_test')

net = torchvision.models.resnet18(pretrained=True)
for param in net.parameters():
    param.requires_grad = False

num_ftrs = net.fc.in_features
net.fc = nn.Sequential(nn.Linear(num_ftrs, num_ftrs),
                       nn.Tanh(), nn.Linear(num_ftrs, num_ftrs),
                       nn.Tanh(), nn.Linear(num_ftrs, num_ftrs),
                       nn.Tanh(), nn.Linear(num_ftrs, num_ftrs),
                       nn.Tanh(), nn.Linear(num_ftrs, num_ftrs),
                       nn.Tanh(), nn.Linear(num_ftrs, len(classes))
                       )