#!/bin/bash
#source: https://gist.github.com/rom1504/474f97a95a526d40ae44a3fc3c657a2e
#SBATCH --job-name=accel_dist
#SBATCH --nodes 2
#SBATCH --ntasks-per-node 1
#SBATCH -c 32
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:59:00
#SBATCH --mem=32G
#SBATCH --output=%x_%j.out
# SBATCH --exclusive



#module load intelmpi
#source /opt/intel/mpi/latest/env/vars.sh
##export LD_LIBRARY_PATH=/opt/aws-ofi-nccl/lib:/opt/amazon/efa/lib64:/usr/local/cuda-11.0/efa/lib:/usr/local/cuda-11.0/lib:/usr/local/cuda-11.0/lib64:/usr/local/cuda-11.0:/opt/nccl/build/lib:/opt/aws-ofi-nccl-install/lib:/opt/aws-ofi-nccl/lib:$LD_LIBRARY_PATH
#export NCCL_PROTO=simple
#export PATH=/opt/amazon/efa/bin:$PATH
#export LD_PRELOAD="/opt/nccl/build/lib/libnccl.so"

#export FI_EFA_FORK_SAFE=1
#export FI_LOG_LEVEL=1
#export FI_EFA_USE_DEVICE_RDMA=1 # use for p4dn

#export NCCL_ALGO=ring
export NCCL_DEBUG=info
#export NCCL_DEBUG_SUBSYS=INIT,ENV,GRAPH,COLL


export PYTHONFAULTHANDLER=1
export CUDA_LAUNCH_BLOCKING=0
export OMPI_MCA_mtl_base_verbose=1
export FI_EFA_ENABLE_SHM_TRANSFER=0
export FI_PROVIDER=efa
export FI_EFA_TX_MIN_CREDITS=64
export NCCL_TREE_THRESHOLD=0


#export NCCL_P2P_DISABLE=1

#export NCCL_IBEXT_DISABLE=1
#export NCCL_SOCKET_IFNAME="eth0,en,eth,em,bond"

# sent to sub script
export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=12802
export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`

echo go $COUNT_NODE
echo $HOSTNAMES

srun ./accelerate.sh
