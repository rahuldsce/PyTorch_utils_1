# transforms.py
from torchvision import transforms


# Train Phase transformations
def mnist_transforms():
  train_transforms = transforms.Compose([
                                        transforms.RandomRotation((-7.0,7.0), fill=(1,)),
                                        transforms.RandomAffine(degrees=20, translate=(0.1,0.1), scale=(0.9, 1.1)),
                                        transforms.ColorJitter(brightness=0.10, contrast=0.1, saturation=0.10, hue=0.1),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,)) # The mean and std have to be sequences (e.g., tuples), therefore you should add a comma after the values. 
                                        ])

  # Test Phase transformations
  test_transforms = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))
                                        ])
  
  return train_transforms, test_transforms


# Train Phase transformations
def cifar10_transforms():
  transform_train = transforms.Compose([
     transforms.RandomCrop(32, padding=4),
     transforms.RandomHorizontalFlip(),
     transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
   ])

  transform_test = transforms.Compose([
     transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
   ])
  
  return transform_train, transform_test
