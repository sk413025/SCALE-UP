#!/bin/bash

# conda activate TomTestEnv

export PYTHONPATH="/home/lab/SCALE-UP:$PYTHONPATH"
cd /home/lab/SCALE-UP
python test_BadNets.py
python torch_model_wrapper.py --model_type=benign
python dataloader2tensor_CIFAR10.py
python torch_model_wrapper.py --model_type=backdoor
python test.py
