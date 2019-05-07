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


# get some random training images
# if __name__ == '__main__':
#    dataiter = iter(trainloader)
#   images, labels = dataiter.next()

# show images
#    imshow(torchvision.utils.make_grid(images))
# print labels
#    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

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

#net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

if __name__ == '__main__':
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

torch.save(net.state_dict(), "transfer-4-tanh.pt")

if __name__ == '__main__':
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
if __name__ == '__main__':
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