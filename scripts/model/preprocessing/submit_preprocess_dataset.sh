#!/bin/bash
#SBATCH -p compute                      # Specify partition [Compute/Memory/GPU]
#SBATCH -N 1 -c 128                     # Specify number of nodes and processors per task
#SBATCH --ntasks-per-node=1		        # Specify number of tasks per node
#SBATCH --gpus-per-node=0		        # Specify total number of GPUs
#SBATCH -t 120:00:00                    # Specify maximum time limit (hour: minute: second)
#SBATCH -A <project_name>               # Specify project name
#SBATCH -J preprocess_dataset           # Specify job name

ml restore
ml Mamba
conda deactivate
conda activate <your_env>

python ./preprocess_dataset.py \
    --tokenizer_name_or_path /path/to/tokenizer \
    --dataset_path /path/to/dataset_1 \
    --output_path /path/to/output \
    --max_sequence_length 2048 \
    --is_local