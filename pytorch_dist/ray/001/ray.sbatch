#!/bin/bash -l
#source: https://docs.ray.io/en/latest/cluster/vms/user-guides/community/slurm.html
#SBATCH --job-name=ray_dist
#SBATCH --nodes 2
#SBATCH --ntasks-per-node 2
#SBATCH -c 32
#SBATCH --gres=gpu:a100:2
#SBATCH --time=00:10:00
#SBATCH --mem=32G
#SBATCH --output=%x_%j.out
# SBATCH --exclusive

set -x #echo on


module purge
module load cesga/system miniconda3/22.11
eval "$(conda shell.bash hook)"
conda deactivate
source $STORE/mytorchdist/bin/activate

nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
if [[ "$head_node_ip" == *" "* ]]; then
IFS=' ' read -ra ADDR <<<"$head_node_ip"
IFS=' ' read -ra ADDR <<<"$head_node_ip"
if [[ ${#ADDR[0]} -gt 16 ]]; then
  head_node_ip=${ADDR[1]}
else
  head_node_ip=${ADDR[0]}
fi
echo "IPV6 address detected. We split the IPV4 address as $head_node_ip"
fi

port=6379
ip_head=$head_node_ip:$port
export ip_head
echo "IP Head: $ip_head"

ray disable-usage-stats

echo "Starting HEAD at $head_node"
./raymaster.sh $head_node $head_node_ip $port
    # optional, though may be useful in certain versions of Ray < 1.0.
    sleep 10
    
    # number of nodes other than the head node
    worker_num=$((SLURM_JOB_NUM_NODES - 1))
    
    for ((i = 1; i <= worker_num; i++)); do
        node_i=${nodes_array[$i]}
        echo "Starting WORKER $i at $node_i"
        ./rayworkers.sh $node_i $ip_head
        sleep 5
    done



which python
python -u ray-train-mnist.py "$SLURM_CPUS_PER_TASK"
