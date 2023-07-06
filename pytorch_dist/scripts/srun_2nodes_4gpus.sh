#/bin/sh

srun -N 2 -n2 --gres=gpu:4 --time=00:59:00 --mem=16G -c 128 $1
