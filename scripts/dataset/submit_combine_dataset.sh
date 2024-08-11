#!/bin/bash
#SBATCH -p compute                          # Specify partition [Compute/Memory/GPU]
#SBATCH -N 1 -c 128                     # Specify number of nodes and processors per task
#SBATCH --ntasks-per-node=1		# Specify number of tasks per node
#SBATCH --gpus-per-node=0		        # Specify total number of GPUs
#SBATCH -t 120:00:00                     # Specify maximum time limit (hour: minute: second)
#SBATCH -A <project_name>                     # Specify project name
#SBATCH -J sample_dataset                          # Specify job name

ml restore
ml Mamba
conda deactivate
conda activate <your_env>

python ./combine_dataset.py \
    --dataset_path_1 /project/lt200258-aithai/boss/refactor/continue-pretraining/example/sample_dataset/sample_dataset \
    --dataset_path_2 /project/lt200258-aithai/boss/refactor/continue-pretraining/example/sample_dataset/sample_dataset \
    --output_path ./combine_dataset  \
    --ratio_1 1. \
    --ratio_2 1. \
    --split train \
    --seed 42 \
    --is_local
