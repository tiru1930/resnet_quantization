import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data.sampler import Sampler
import numpy as np
import os 

NUM_WORKERS = 2


def get_imagenet(dataset_dir='data_set_path', batch_size=32):
    
    trainset,testset = imagenet_get_datasets(dataset_dir)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=NUM_WORKERS,
                                              pin_memory=True, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=NUM_WORKERS,
                                             pin_memory=True, shuffle=False)
    return trainloader, testloader


class MyImageFolder(datasets.ImageFolder):
    """docstring for ClassName"""
    def __getitem__(self, index):
        return super(MyImageFolder, self).__getitem__(index), self.imgs[index]

def imagenet_get_datasets(data_dir):
    """
    Load the ImageNet dataset.
    """
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
        #transforms.Resize(256),
        transforms.RandomResizedCrop(256),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = MyImageFolder(train_dir, train_transform)

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        normalize,
    ])

    test_dataset = MyImageFolder(test_dir, test_transform)
    return train_dataset, test_dataset
        
        

if __name__ == "__main__":
    # print("CIFAR10")
    # print(get_cifar(10))
    # print("---"*20)
    # print("---"*20)
    # print("CIFAR100")
    # print(get_cifar(100))

    print("IMAGENET")
    print(get_imagenet())
