#!/bin/bash

export PYTHONPATH="~/SCALE-UP:$PYTHONPATH"
cd ~/SCALE-UP
# python test_BadNets_cifar10.py

# newest_folder=$(ls -dt experiments/train_poisoned_CIFAR10_*/ | head -1)
# cp "${newest_folder}"ckpt_epoch_100.pth badnet/

# python torch_model_wrapper.py --model_type=benign
# python dataloader2tensor_CIFAR10.py
# python torch_model_wrapper.py --model_type=backdoor
# python test.py


# python test_WaNet_cifar10.py

# newest_folder=$(ls -dt experiments/train_poisoned_CIFAR10_*/ | head -1)
# cp "${newest_folder}"ckpt_epoch_100.pth badnet/

# python torch_model_wrapper.py --model_type=benign
# python dataloader2tensor_CIFAR10.py
# python torch_model_wrapper.py --model_type=backdoor
# python test.py
