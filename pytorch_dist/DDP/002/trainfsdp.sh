#!/bin/bash

#module reset
#module load miniconda3
#conda activate mytorch
#srun python mnist_classify_ddp.py --epochs=2
#source $STORE/mytorchdist/bin/activate
#conda activate mytorchdist
source $STORE/conda/envs/mytorchdist/bin/activate
python fsdp.py --epochs=2
#torchrun mnist_classify_ddp.py --epochs=2

