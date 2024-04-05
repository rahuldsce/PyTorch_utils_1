# dataset.py

import numpy as np

import albumentations as A
from albumentations.pytorch import ToTensorV2

from torch.utils.data import Dataset
from torchvision import datasets

from . import transforms
from . import datasetmgr

def mnist():
  train_transforms, test_transforms = transforms.mnist_transforms()
  train = datasets.MNIST('./data', train=True, download=True, transform=train_transforms)
  test = datasets.MNIST('./data', train=False, download=True, transform=test_transforms)
  return train, test


def cifar10():
  train_transforms, test_transforms = transforms.cifar10_transforms()
  train = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transforms)
  test = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transforms)
  return train, test

class CIFAR10Sequence(Dataset):
    def randomPadding(self, x):
      return np.pad(x, ((4,4), (4,4), (0,0)), mode='constant')
    
    def __init__(self, x_set, y_set, transform=None):
      self.x = x_set
      self.y = y_set
      self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
      image = self.transform(image=self.randomPadding(self.x[index]))["image"]
      return image, self.y[index]


def cifar10_albumentations():
    AUGMENTATIONS_TRAIN = A.Compose([
                                A.HorizontalFlip(p=0.5),
                                A.ShiftScaleRotate(),
                                A.CoarseDropout(max_holes=1, max_height=16, max_width=16,  # Corrected parameters
                                                min_holes=1, min_height=16, min_width=16,   # Corrected parameters
                                                fill_value=(0.5, 0.5, 0.5), mask_fill_value=None),
                                A.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
                                ToTensorV2()
                                ])

    AUGMENTATIONS_TEST = A.Compose([
                                A.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
                                ToTensorV2()
                                ])

    data_url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    img_size = 32
    num_channels = 3
    num_classes = 10

    cifar10DataMgr = datasetmgr.DatasetManager(data_url, "cifar-10-data", "cifar-10-batches-py", img_size, num_channels, num_classes)

    # Download and extract CIFAR-10 data
    cifar10DataMgr.maybe_download_and_extract()

    # training data
    x_train, y_train = cifar10DataMgr.load_training_data()
    train = CIFAR10Sequence(x_train, y_train, transform=AUGMENTATIONS_TRAIN)

    # Validation data
    x_val, y_val = cifar10DataMgr.load_validation_data(5000)
    test = CIFAR10Sequence(x_val, y_val, transform=AUGMENTATIONS_TEST)

    return train, test

def cifar10_s10_albumentations():
    AUGMENTATIONS_TRAIN = A.Compose([
                                A.RandomCrop(32, 32, padding=4),
                                A.HorizontalFlip(),
                                A.CoarseDropout(max_holes=1, max_height=8, max_width=8,  # Corrected parameters
                                                min_holes=1, min_height=8, min_width=8,   # Corrected parameters
                                                fill_value=(0.5, 0.5, 0.5), mask_fill_value=None),
                                A.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
                                ToTensorV2()
                                ])

    AUGMENTATIONS_TEST = A.Compose([
                                A.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
                                ToTensorV2()
                                ])

    data_url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    img_size = 32
    num_channels = 3
    num_classes = 10

    cifar10DataMgr = datasetmgr.DatasetManager(data_url, "cifar-10-data", "cifar-10-batches-py", img_size, num_channels, num_classes)

    # Download and extract CIFAR-10 data
    cifar10DataMgr.maybe_download_and_extract()

    # training data
    x_train, y_train = cifar10DataMgr.load_training_data()
    train = CIFAR10Sequence(x_train, y_train, transform=AUGMENTATIONS_TRAIN)

    # Validation data
    x_val, y_val = cifar10DataMgr.load_validation_data(5000)
    test = CIFAR10Sequence(x_val, y_val, transform=AUGMENTATIONS_TEST)

    return train, test
