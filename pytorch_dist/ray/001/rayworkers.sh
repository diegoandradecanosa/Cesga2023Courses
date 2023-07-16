#!/bin/bash



node_i=${1}
ip_head=${2}

srun --nodes=1 --ntasks=1 -w "$node_i" \
            ray start --address "$ip_head" \
            --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "${SLURM_GPUS_ON_NODE}" --block &
