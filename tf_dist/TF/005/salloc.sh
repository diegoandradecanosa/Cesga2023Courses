#!/bin/bash

source $STORE/mytf/bin/activate

export LD_LIBRARY_PATH=$STORE/mytf/lib/python3.9/site-packages/nvidia/cuda_cupti/lib/:$LD_LIBRARY_PATH
which python
salloc -N 2 --ntasks-per-node=2 --gres=gpu:a100:2 -c 32 --mem=32G -t 00:59:00 -p short --qos=short
srun python $1 -ps 1 -workers 3

