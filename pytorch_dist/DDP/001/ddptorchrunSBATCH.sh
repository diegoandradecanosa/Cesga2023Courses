#!/bin/bash
# Based on: https://github.com/PrincetonUniversity/multi_gpu_training/tree/main/02_pytorch_ddp
#SBATCH --job-name=ddp-torch     # create a short name for your job
#SBATCH --nodes=2  
#SBATCH --ntasks-per-node=2     
#SBATCH -c 32 
#SBATCH --mem=128G                
#SBATCH --gres=gpu:a100:2        # number of GPUs per node          
#SBATCH --time=00:05:00          # total run time limit (HH:MM:SS)

source $STORE/conda/envs/mytorchdist/bin/activate
srun ./train.sh $HOSTNAME