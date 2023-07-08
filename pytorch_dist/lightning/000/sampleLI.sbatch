#!/bin/bash -l

# SLURM SUBMIT SCRIPT
#SBATCH -N 2             # This needs to match Trainer(num_nodes=...)
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks-per-node=2   # This needs to match Trainer(devices=...)
#SBATCH --mem=8G
#SBATCH -c 32
#SBATCH --time=0-00:10:00

source $STORE/conda/envs/mytorchdist/bin/activate

# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

pythonint=$(which python)
srun python sampleLI.py

# Based on: https://pytorch-lightning.readthedocs.io/en/stable/clouds/cluster_advanced.html