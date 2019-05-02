import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class ImageFolderWithPaths(datasets.ImageFolder):
    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = ImageFolderWithPaths(root='Data/intel/seg_train', transform=transform)
# trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
#                                        download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=1, shuffle=True, num_workers=2)

testset = ImageFolderWithPaths(root='Data/intel/seg_test', transform=transform)
# testset = torchvision.datasets.CIFAR10(root='./data', train=False,
#                                       download=True, transform=transform)
testloader = DataLoader(testset, batch_size=1, shuffle=True, num_workers=2)

classes = os.listdir('Data/intel/seg_test')
# classes = ('plane', 'car', 'bird', 'cat',
#           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

removed_stem = 'Data/intel/removed'
rm_test = removed_stem + '/rm_test'
rm_train = removed_stem + '/rm_train'


if __name__ == '__main__':
    for i, data in enumerate(trainloader, 0):
        inputs, labels, path = data
        if inputs.shape[2] != 150 or inputs.shape[3] != 150:
            path_split = path[0].split('\\')
            source_path = path_split[0] + '/' + path_split[1] + '/' + path_split[2]
            destination = rm_train + '/' + path_split[1] + '/' + path_split[2]
            os.rename(source_path, destination)

if __name__ == '__main__':
    for i, data in enumerate(testloader, 0):
        inputs, labels, path = data
        if inputs.shape[2] != 150 or inputs.shape[3] != 150:
            path_split = path[0].split('\\')
            source_path = path_split[0] + '/' + path_split[1] + '/' + path_split[2]
            destination = rm_test + '/' + path_split[1] + '/' + path_split[2]
            os.rename(source_path, destination)

