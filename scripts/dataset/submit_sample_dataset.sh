#!/bin/bash
#SBATCH -p compute                          # Specify partition [Compute/Memory/GPU]
#SBATCH -N 1 -c 128                         # Number of nodes and processors per task
#SBATCH --ntasks-per-node=1                 # Number of tasks per node
#SBATCH --gpus-per-node=0                   # Number of GPUs
#SBATCH -t 120:00:00                        # Maximum time limit (hour: minute: second)
#SBATCH -A <project_name>                   # Project name
#SBATCH -J sample_dataset                   # Job name

ml restore
ml Mamba
conda deactivate
conda activate <your_env>

python ./sample_dataset.py \
    --dataset_path /path/to/dataset \
    --output_path /path/to/output \
    --ratio 0.8 \
    --seed 42 \
    --is_local
