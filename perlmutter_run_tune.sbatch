#!/bin/bash
#SBATCH -A m4431_g    
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH -G 4
#SBATCH -J raytune_pathmnist
#SBATCH -o logs/out.%j.txt
#SBATCH -e logs/err.%j.txt

mkdir -p logs
mkdir -p /global/u2/y/yuning88/rl-hyper-tuning/ray_results

module load pytorch


cd ~/rl-hyper-tuning

srun -n 1 python train_tune.py
