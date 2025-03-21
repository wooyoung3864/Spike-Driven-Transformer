#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate spike_driven

# Print Python environment details
echo "Python executable: $(which python)"
echo "Python version: $(python --version)"
echo "Python site-packages: $(python -c 'import sys; print(sys.path)')"

export PYTHONPATH=/home/wyjung/miniconda3/envs/spike_driven/lib/python3.9/site-packages:$PYTHONPATH


CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29501 train.py -c conf/cifar10/2_256_300E_t4.yml --model sdt --spike-mode lif
