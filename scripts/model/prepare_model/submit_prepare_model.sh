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

python ./prepare_model.py \
    --model_name_or_path <model_name_or_path> \
    --tokenizer_path <path_to_tokenizer>\
    --output_path ./resize_model \
