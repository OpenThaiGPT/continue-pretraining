#!/bin/bash
#SBATCH -p compute                      # Specify partition [Compute/Memory/GPU]
#SBATCH -N 1 -c 128                     # Specify number of nodes and processors per task
#SBATCH --ntasks-per-node=1		        # Specify number of tasks per node
#SBATCH --gpus-per-node=0		        # Specify total number of GPUs
#SBATCH -t 120:00:00                    # Specify maximum time limit (hour: minute: second)
#SBATCH -A <project_name>               # Specify project name
#SBATCH -J merge_tokenizer              # Specify job name

ml restore
ml Mamba
conda deactivate
conda activate <your_env>

python ./merge.py \
    --main_tokenizer_path <path_to_original_llama_tokenizer> \
    --main_tokenizer_path <path_to_extra_sentencepiece_tokenizer> \
    --output_path <path_to_save>  \