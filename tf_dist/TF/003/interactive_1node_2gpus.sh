#/bin/sh
srun --gres=gpu:2 --time=00:59:00 --mem=16G -c 64 --pty bash
