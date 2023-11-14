#!/bin/sh
#SBATCH -J paramser       # Job name
#SBATCH -o paramser.o%j   # Name of stdout output file(%j expands to jobId)
#SBATCH -e paramser.e%j   # Name of stderr output file(%j expands to jobId)
#SBATCH -N 2
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:a100:2
#SBATCH -c 32
#SBATCH --mem=32G
#SBATCH -t 00:59:00
#SBATCH -p short --qos=short

#module load tensorflow/2.5.0-cuda-system
source $STORE/mytf/bin/activate

echo SLURM_NTASKS: $SLURM_NTASKS
echo SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK
echo SLURM_NNODES: $SLURM_NNODES
echo SLURM_NTASKS_PER_NODE: $SLURM_NTASKS_PER_NODE

PS=1
WORKERS=$((SLURM_NTASKS-PS))

echo PS: $PS
echo WORKERS: $WORKERS

export LD_LIBRARY_PATH=$STORE/mytf/lib/python3.9/site-packages/nvidia/cuda_cupti/lib/:$LD_LIBRARY_PATH
which python
#salloc -N2 --ntasks-per-node=2 --gres=gpu:a100:2 -c 32 --mem=32G -t 00:59:00 -p short --qos=short
#srun python $1 -ps $PS -workers $WORKERS


#echo "srun -N $SLURM_NNODES --ntasks-per-node $SLURM_NTASKS_PER_NODE -c $SLURM_CPUS_PER_TASK  --resv-ports=$SLURM_NTASKS_PER_NODE -l python $1 -ps $PS -workers $WORKERS"
#srun -N $SLURM_NNODES --ntasks-per-node $SLURM_NTASKS_PER_NODE -c $SLURM_CPUS_PER_TASK  --resv-ports=$SLURM_NTASKS_PER_NODE -l python $1 -ps $PS -workers $WORKERS
#srun python $1 -ps $PS -workers $WORKERS

srun python $1