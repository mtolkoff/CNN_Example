import torch
import torchvision

from torch.utils.data import DataLoader
from torchvision import transforms, datasets

import os


torch.manual_seed(123)
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


class ImageFolderWithPaths(datasets.ImageFolder):
    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


trainset = ImageFolderWithPaths(root='Data/intel/removed/rm_train', transform=transform)
trainloader = DataLoader(trainset, batch_size=1, shuffle=True, num_workers=2)

testset = ImageFolderWithPaths(root='Data/intel/removed/rm_test', transform=transform)
testloader = DataLoader(testset, batch_size=1, shuffle=True, num_workers=2)

OLD_PATH = 'Data/intel/seg_train'
OLD_PATH_TEST = 'Data/intel/seg_test'


if __name__ == '__main__':
    for i, item in enumerate(trainloader, 0):
        images, labels, path = item
        imageCat = torch.zeros((1, 3, 150 - images.shape[2], 150))
        #imageCat.new_empty((1, 3, 150 - images.shape[2], 150))
        image_reshaped = torch.zeros(1, 3, 150, 150)
        torch.cat((images, imageCat), dim=2, out=image_reshaped)
        return_path = OLD_PATH + '/' + path[0].split('\\')[1] + '/' + path[0].split('\\')[2]
        print(OLD_PATH + '/' + path[0].split('\\')[1] + '/' + path[0].split('\\')[2])
        torchvision.utils.save_image(image_reshaped, return_path)


    for i, item in enumerate(testloader, 0):
        images, labels, path = item
        imageCat = torch.zeros((1, 3, 150 - images.shape[2], 150))
        # imageCat.new_empty((1, 3, 150 - images.shape[2], 150))
        image_reshaped = torch.zeros(1, 3, 150, 150)
        torch.cat((images, imageCat), dim=2, out=image_reshaped)
        return_path = OLD_PATH_TEST + '/' + path[0].split('\\')[1] + '/' + path[0].split('\\')[2]
        print(OLD_PATH_TEST + '/' + path[0].split('\\')[1] + '/' + path[0].split('\\')[2])
        torchvision.utils.save_image(image_reshaped, return_path)