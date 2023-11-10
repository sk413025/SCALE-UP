#!/bin/bash

# conda activate TomTestEnv

export PYTHONPATH="/home/lab/SCALE-UP:$PYTHONPATH"
cd /home/lab/SCALE-UP
rm badnet/*

python test_BadNets_cifar10.py

newest_folder=$(ls -dt experiments/train_poisoned_CIFAR10_*/ | head -1)
cp "${newest_folder}"ckpt_epoch_200.pth badnet/

python torch_model_wrapper.py --model_type=benign
python dataloader2tensor_CIFAR10.py
python torch_model_wrapper.py --model_type=backdoor
python test.py
