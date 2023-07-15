#!/bin/bash
echo myuser=`whoami`
echo COUNT_NODE=$COUNT_NODE
echo LD_LIBRARY_PATH = $LD_LIBRARY_PATH
echo PATH = $PATH
echo which mpicc `which mpicc`
echo HOSTNAMES = $HOSTNAMES
echo hostname = `hostname --alias`
echo MASTER_ADDR= $MASTER_ADDR
echo MASTER_PORT= $MASTER_PORT

H=`hostname --alias`
THEID=`echo -e $HOSTNAMES  | python3 -c "import sys;[sys.stdout.write(str(i)) for i,line in enumerate(next(sys.stdin).split(' ')) if line.strip() == '$H'.strip()]"`
echo THEID=$THEID

#module load miniconda3
# conda activate myacc
#source $STORE/conda/envs/mytorchdist/bin/activate
module purge
module load cesga/system miniconda3/22.11
eval "$(conda shell.bash hook)"
source $STORE/conda/envs/mytorchdist/bin/activate

export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
#accelerate launch --multi_gpu --num_machines 2 --num_processes 2 --num_cpu_threads_per_process 32 accelerate_sample.py
which accelerate
accelerate launch --num_processes $(( 1 * $COUNT_NODE )) --num_machines $COUNT_NODE --multi_gpu --mixed_precision fp16 --machine_rank $THEID --main_process_ip $MASTER_ADDR --main_process_port $MASTER_PORT  accelerate_sample.py  --logging_dir logs

echo "DONE¡"
