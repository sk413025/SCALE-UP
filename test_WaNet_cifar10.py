'''
This is the test code of benign training and poisoned training on torchvision.datasets.CIFAR10.
Attack method is BadNets.

https://github.com/THUYimingLi/BackdoorBox/blob/4b0f1b0fba0a921654e443c7ef7dbf738fbfb52b/tests/test_WaNet.py#L276
'''


import os

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision
from torchvision.transforms import Compose, ToTensor, PILToTensor, RandomHorizontalFlip

import core


global_seed = 555
deterministic = True

def gen_grid(height, k):
    """Generate an identity grid with shape 1*height*height*2 and a noise grid with shape 1*height*height*2
    according to the input height ``height`` and the uniform grid size ``k``.
    """
    ins = torch.rand(1, 2, k, k) * 2 - 1
    ins = ins / torch.mean(torch.abs(ins))  # a uniform grid
    noise_grid = nn.functional.upsample(ins, size=height, mode="bicubic", align_corners=True)
    noise_grid = noise_grid.permute(0, 2, 3, 1)  # 1*height*height*2
    array1d = torch.linspace(-1, 1, steps=height)  # 1D coordinate divided by height in [-1, 1]
    x, y = torch.meshgrid(array1d, array1d)  # 2D coordinates height*height
    identity_grid = torch.stack((y, x), 2)[None, ...]  # 1*height*height*2

    return identity_grid, noise_grid


# dataset = torchvision.datasets.MNIST
dataset = torchvision.datasets.CIFAR10

transform_train = Compose([
    ToTensor(),
    RandomHorizontalFlip()
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

directory = 'warnet'
if not os.path.exists(directory):
    os.makedirs(directory)
torch.save(test_samples, os.path.join(directory, 'benign_test_samples.pth'))
torch.save(test_labels, os.path.join(directory, 'benign_labels.pth'))


identity_grid,noise_grid=gen_grid(32,4)
# torch.save(identity_grid, 'ResNet-18_CIFAR-10_WaNet_identity_grid.pth')
# torch.save(noise_grid, 'ResNet-18_CIFAR-10_WaNet_noise_grid.pth')
wanet = core.WaNet(
    train_dataset=trainset,
    test_dataset=testset,
    model=core.models.ResNet(18),
    # model=core.models.BaselineMNISTNetwork(),
    loss=nn.CrossEntropyLoss(),
    y_target=0,
    poisoned_rate=0.1,
    identity_grid=identity_grid,
    noise_grid=noise_grid,
    noise=False,
    seed=global_seed,
    deterministic=deterministic
)

# poisoned_train_dataset, poisoned_test_dataset = wanet.get_poisoned_dataset()

# Train Infected Model
schedule = {
    'device': 'GPU',
    'CUDA_VISIBLE_DEVICES': '0',
    'GPU_num': 1,

    'benign_training': False,
    'batch_size': 128,
    'num_workers': 4,

    'lr': 0.1,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'gamma': 0.1,
    'schedule': [150, 180],

    'epochs': 200,

    'log_iteration_interval': 100,
    'test_epoch_interval': 10,
    'save_epoch_interval': 10,

    'save_dir': 'experiments',
    'experiment_name': 'ResNet-18_CIFAR-10_WaNet'
}

wanet.train(schedule)
infected_model = wanet.get_model()

poisoned_train_dataset, poisoned_test_dataset = wanet.get_poisoned_dataset()
torch.save(poisoned_test_dataset, os.path.join(directory, 'poisoned_test_dataset_WaNet.pth'))

