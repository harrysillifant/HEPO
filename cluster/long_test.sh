#!/bin/bash

# Job Flags
#SBATCH -p mit_normal_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=6:00:00

source /home/hcs/.bashrc

export WANDB_MODE=offline

uv run ppo_hb.py --seed 0 --env_name h1hand-sit_simple-v0 --device cuda --num_envs 32 --name engaging_long_test
