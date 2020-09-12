
# dataset.py

from torchvision import datasets
import transforms

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
