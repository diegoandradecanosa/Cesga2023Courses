#!/bin/bash

#source $STORE/conda/envs/mytorchdist/bin/activate
module purge
module load cesga/system miniconda3/22.11
eval "$(conda shell.bash hook)"
source $STORE/conda/envs/mytorchdist/bin/activate

python mnist.py
