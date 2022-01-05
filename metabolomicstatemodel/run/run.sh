#! /bin/sh
echo $(hostname)
echo $(which python)
echo $(python -c 'import torch; print(f"found {torch.cuda.device_count()} gpus.")')
echo $CUDA_VISIBLE_DEVICES

cd ..
python run/train.py --config-dir run/config/ --config-name config
