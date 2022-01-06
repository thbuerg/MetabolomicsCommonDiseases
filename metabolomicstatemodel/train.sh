#! /bin/sh
echo $(hostname)
echo $(which python)
echo $(python -c 'import torch; print(f"found {torch.cuda.device_count()} gpus.")')
echo $CUDA_VISIBLE_DEVICES

python train.py --config-dir source/config/ --config-name config
