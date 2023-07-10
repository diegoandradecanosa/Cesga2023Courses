#!/bin/bash -l

# SLURM SUBMIT SCRIPT
#SBATCH --nodes=2               # This needs to match Fabric(num_nodes=...)
#SBATCH --ntasks-per-node=2     # This needs to match Fabric(devices=...)
#SBATCH --gres=gpu:a100:2            # Request N GPUs per machine
#SBATCH --mem=32G               # Request 32 GB per node
#SBATCH -c 32                   # Request 32 cores per node (total 64 cores)
#SBATCH --time=0-02:00:00


# Debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

# On your cluster you might need this:
# export NCCL_SOCKET_IFNAME=^docker0,lo

# Run your training script
srun mnist.sh