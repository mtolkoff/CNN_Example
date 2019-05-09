import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

import os

import Models.basic_cnn
import Models.resnet18_pretrained

if __name__ == '__main__':
    torch.manual_seed(123)
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    testset = datasets.ImageFolder(root='Data/intel/seg_test', transform=transform)
    testloader = DataLoader(testset, batch_size=4, shuffle=True, num_workers=2)

    classes = os.listdir('Data/intel/seg_test')

    net = Models.resnet18_pretrained.net
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
