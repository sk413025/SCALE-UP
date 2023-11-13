'''
This is the test code of benign training and poisoned training on torchvision.datasets.CIFAR10.
Attack method is BadNets.
https://github.com/THUYimingLi/BackdoorBox/blob/main/tests/test_cifar10.py
'''


import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision
from torchvision.datasets import DatasetFolder
from torchvision.transforms import Compose, ToTensor, ToPILImage, PILToTensor, RandomHorizontalFlip, Resize

import core

# dataset = torchvision.datasets.MNIST
# dataset = torchvision.datasets.CIFAR10
# dataset = torchvision.datasets.GTSRB

# transform_train = Compose([
#     Resize((32, 32)),
#     ToTensor(),
#     # RandomHorizontalFlip()
# ])
# trainset = dataset('data', split='train', transform=transform_train, download=True)

# transform_test = Compose([
#     Resize((32, 32)),
#     ToTensor()
# ])
# testset = dataset('data', split='test', transform=transform_test, download=True)
datasets_root_dir = '/home/iis519409/SCALE-UP/data'

transform_train = Compose([
    ToPILImage(),
    Resize((32, 32)),
    ToTensor()
])
trainset = DatasetFolder(
    root=os.path.join(datasets_root_dir, 'gtsrb', 'train'), # please replace this with path to your training set
    loader=cv2.imread,
    extensions=('png',),
    transform=transform_train,
    target_transform=None,
    is_valid_file=None)

transform_test = Compose([
    ToPILImage(),
    Resize((32, 32)),
    ToTensor()
])
testset = DatasetFolder(
    root=os.path.join(datasets_root_dir, 'gtsrb', 'testset'), # please replace this with path to your test set
    loader=cv2.imread,
    extensions=('png',),
    transform=transform_test,
    target_transform=None,
    is_valid_file=None)


test_loader = torch.utils.data.DataLoader(
        dataset=testset,
        batch_size=1,
        shuffle=False)

test_samples = torch.zeros((10000, 3, 32, 32))
test_labels = torch.zeros((10000, 1))

for batch_id, batch in enumerate(test_loader):
    if batch_id >= 10000:
        break
    batch_img = batch[0]
    batch_label = batch[1]
    test_samples[batch_id, :, :, :] = batch_img
    test_labels[batch_id] = batch_label
    if (batch_id + 1) % 100 == 0:
        print((batch_id + 1)/100)


directory = 'badnet-gtsrb'
if not os.path.exists(directory):
    os.makedirs(directory)
torch.save(test_samples, os.path.join(directory, 'benign_test_samples.pth'))
torch.save(test_labels, os.path.join(directory, 'benign_labels.pth'))

pattern = torch.zeros((32, 32), dtype=torch.uint8)
pattern[-3:, -3:] = 255
weight = torch.zeros((32, 32), dtype=torch.float32)
weight[-3:, -3:] = 1.0

# import ipdb; ipdb.set_trace()
# badnets = core.BadNets(
#     train_dataset=trainset,
#     test_dataset=testset,
#     model=core.models.ResNet(18, 43),
#     # model=core.models.BaselineMNISTNetwork(),
#     loss=nn.CrossEntropyLoss(),
#     y_target=0,
#     poisoned_rate=0.20,
#     pattern=pattern,
#     weight=weight,
#     seed=666
# )
global_seed = 666
deterministic = True

badnets = core.BadNets(
    train_dataset=trainset,
    test_dataset=testset,
    model=core.models.ResNet(18, 43),
    loss=nn.CrossEntropyLoss(),
    y_target=1,
    poisoned_rate=0.05,
    pattern=pattern,
    weight=weight,
    poisoned_transform_train_index=2,
    poisoned_transform_test_index=2,
    seed=global_seed,
    deterministic=deterministic
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
#     'experiment_name': 'train_benign_GTSRB'
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
    'schedule': [20],

    'epochs': 100,

    'log_iteration_interval': 50,
    'test_epoch_interval': 10,
    'save_epoch_interval': 10,

    'save_dir': 'experiments',
    'experiment_name': 'train_poisoned_GTSRB'
}

badnets.train(schedule)

poisoned_train_dataset, poisoned_test_dataset = badnets.get_poisoned_dataset()
torch.save(poisoned_test_dataset, os.path.join(directory, 'poisoned_test_dataset_BadNets.pth'))


