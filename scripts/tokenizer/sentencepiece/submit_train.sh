#!/bin/bash
#SBATCH -p memory                       # Specify partition [Compute/Memory/GPU]
#SBATCH -N 1 -c 128                     # Specify number of nodes and processors per task
#SBATCH --ntasks-per-node=1		        # Specify number of tasks per node
#SBATCH --gpus-per-node=0		        # Specify total number of GPUs
#SBATCH -t 120:00:00                    # Specify maximum time limit (hour: minute: second)
#SBATCH -A <project_name>               # Specify project name
#SBATCH -J sentencepiece_training       # Specify job name

ml restore
ml Mamba
conda deactivate
conda activate <your_env>

python ./train.py \
    --output_path /path/to/output \
    --vocab_size 32000 \
    --load_dataset_path /path/to/dataset \
    --mode bpe \
    --large_corpus \
    --is_local
