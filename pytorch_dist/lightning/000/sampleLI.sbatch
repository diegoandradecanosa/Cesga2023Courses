#!/bin/bash -l
# SLURM SUBMIT SCRIPT
#SBATCH -N 2             # This needs to match Trainer(num_nodes=...)
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks-per-node=2   # This needs to match Trainer(devices=...)
#SBATCH --mem=8G
#SBATCH -c 32
#SBATCH --time=0-00:10:00

module purge
module load cesga/system miniconda3/22.11
eval "$(conda shell.bash hook)"
conda deactivate
source $STORE/mytorchdist/bin/deactivate
source $STORE/mytorchdist/bin/activate

# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

which python
srun python sampleLI.py

# Based on: https://pytorch-lightning.readthedocs.io/en/stable/clouds/cluster_advanced.html
