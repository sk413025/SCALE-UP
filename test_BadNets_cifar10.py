'''
This is the test code of benign training and poisoned training on torchvision.datasets.CIFAR10.
Attack method is BadNets.
https://github.com/THUYimingLi/BackdoorBox/blob/main/tests/test_cifar10.py
'''


import os

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision
from torchvision.transforms import Compose, ToTensor, PILToTensor, RandomHorizontalFlip

import core

# dataset = torchvision.datasets.MNIST
dataset = torchvision.datasets.CIFAR10


transform_train = Compose([
    ToTensor(),
    # RandomHorizontalFlip()
])
trainset = dataset('data', train=True, transform=transform_train, download=True)

transform_test = Compose([
    ToTensor()
])
testset = dataset('data', train=False, transform=transform_test, download=True)

test_loader = torch.utils.data.DataLoader(
        dataset=testset,
        batch_size=1,
        shuffle=False)


test_samples = torch.zeros((10000, 3, 32, 32))
test_labels = torch.zeros((10000, 1))
for batch_id, batch in enumerate(test_loader):
    batch_img = batch[0]
    batch_label = batch[1]
    test_samples[batch_id, :, :, :] = batch_img
    test_labels[batch_id] = batch_label
    if (batch_id + 1) % 100 == 0:
        print((batch_id + 1)/100)

directory = 'badnet'
if not os.path.exists(directory):
    os.makedirs(directory)
torch.save(test_samples, os.path.join(directory, 'benign_test_samples.pth'))
torch.save(test_labels, os.path.join(directory, 'benign_labels.pth'))

badnets = core.BadNets(
    train_dataset=trainset,
    test_dataset=testset,
    model=core.models.ResNet(18),
    # model=core.models.BaselineMNISTNetwork(),
    loss=nn.CrossEntropyLoss(),
    y_target=0,
    poisoned_rate=0.20,
    seed=666
)

# # train benign model
# schedule = {
#     'device': 'GPU',
#     'CUDA_VISIBLE_DEVICES': '0',
#     'GPU_num': 1,

#     'benign_training': True,
#     'batch_size': 128,
#     'num_workers': 16,

#     'lr': 0.1,
#     'momentum': 0.9,
#     'weight_decay': 5e-4,
#     'gamma': 0.1,
#     'schedule': [150, 180],

#     'epochs': 200,

#     'log_iteration_interval': 100,
#     'test_epoch_interval': 10,
#     'save_epoch_interval': 10,

#     'save_dir': 'experiments',
#     'experiment_name': 'train_benign_CIFAR10'
# }

# badnets.train(schedule)

# train Infected model
schedule = {
    'device': 'GPU',
    'CUDA_VISIBLE_DEVICES': '0',
    'GPU_num': 1,

    'benign_training': False,
    'batch_size': 128,
    'num_workers': 16,

    'lr': 0.1,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'gamma': 0.1,
    'schedule': [150, 180],

    'epochs': 100,

    'log_iteration_interval': 50,
    'test_epoch_interval': 10,
    'save_epoch_interval': 10,

    'save_dir': 'experiments',
    'experiment_name': 'train_poisoned_CIFAR10'
}

badnets.train(schedule)

poisoned_train_dataset, poisoned_test_dataset = badnets.get_poisoned_dataset()
torch.save(poisoned_test_dataset, os.path.join(directory, 'poisoned_test_dataset_BadNets.pth'))


