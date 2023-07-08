#!/bin/bash

TASKS_PER_NODE=$(( SLURM_NTASKS / SLURM_JOB_NUM_NODES ))
GPUS_PER_NODE=$(nvidia-smi --list-gpus | wc -l)
MASTER=$1
export MASTER_PORT=$((SLURM_JOBID%10000+20000))

echo "torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes $SLURM_JOB_NUM_NODES --node_rank $SLURM_NODEID --rdzv_id=$SLURM_JOB_ID --rdzv_endpoint=$MASTER:12373 --master_addr $MASTER mnist_classify_ddp.py --epochs=2"
torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes $SLURM_JOB_NUM_NODES --node_rank $SLURM_NODEID --rdzv_id=$SLURM_JOB_ID --rdzv_endpoint=$MASTER:$MASTER_PORT --master_addr $MASTER mnist_classify_ddp.py --epochs=2
