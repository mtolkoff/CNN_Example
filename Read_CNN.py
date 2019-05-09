import torch

import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import transforms, datasets

import os
import matplotlib.pyplot as plt
import numpy as np

import Models.resnet18_pretrained
import Models.basic_cnn

if __name__ == '__main__':
    torch.manual_seed(123)
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = datasets.ImageFolder(root='Data/intel/seg_train', transform=transform)
    trainloader = DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

    classes = os.listdir('Data/intel/seg_test')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# classes = ('plane', 'car', 'bird', 'cat',
#           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# if __name__ == '__main__':
#     for i, data in enumerate(trainloader, 0):
#         inputs, labels, path = data
#         if inputs.shape[2] != 150 or inputs.shape[3] != 150:
#             print(path)
# functions to show an image
    def imshow(img):
        img = img / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

#Import from models folder. Change this line to change the model.
    net = Models.resnet18_pretrained.net

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


    for epoch in range(1):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
 #           inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 500 == 499:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.5f' %
                      (epoch + 1, i + 1, running_loss / 500))
                running_loss = 0.0

    print('Finished Training')
