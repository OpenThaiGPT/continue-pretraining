# Dataset Processing
Contains scripts to combine and sample datasets using the HuggingFace datasets library. The processes are designed to be run on a computing cluster using SLURM.

## Usage

### Combining Datasets
The `combine_dataset.py` script combines two datasets by sampling a ratio of each dataset and concatenating them.

#### Command-line Arguments
- `--dataset_path_1`: Path to the first dataset (required).
- `--dataset_path_2`: Path to the second dataset (required).
- `--output_path`: Path to save the combined dataset (required).
- `--is_local`: Whether the datasets are loaded from local files (optional, default: False).
- `--ratio_1`: Sampling ratio for the first dataset based on a smaller dataset (optional, default: 1.0).
- `--ratio_2`: Sampling ratio for the second dataset based on a smaller dataset (optional, default: 1.0).
- `--split`: Dataset split to load (e.g., train, test, validation) (optional, default: train).
- `--seed`: Seed for random shuffling (optional, default: 42).
- `--buffer_size`: Buffer for random shuffling (optional, default: 1000).
- `--num_proc`: Number of processes for tokenization (optional, default: number of CPU cores).

#### Example Usage
```bash
python combine_dataset.py \
    --dataset_path_1 /path/to/dataset_1 \
    --dataset_path_2 /path/to/dataset_2 \
    --output_path /path/to/output \
    --ratio_1 1.0 \
    --ratio_2 1.0 \
    --split train \
    --seed 42 \
    --is_local
```
> Note: A ratio of 1.0 indicates the full size of the smaller dataset.


### Sampling Datasets
The `sample_dataset.py` script samples a portion of the dataset based on the given ratio and saves the sampled dataset.

#### Command-line Arguments
- `--dataset_path`: Path to the dataset (required).
- `--output_path`: Path to save the sampled dataset (required).
- `--is_local`: Whether the dataset is loaded from a local file (optional, default: False).
- `--ratio`: Sampling ratio (optional, default: 0.8).
- `--seed`: Seed for random shuffling (optional, default: 42).
- `--num_proc`: Number of processes for tokenization (optional, default: number of CPU cores).

#### Example Usage
```bash
python sample_dataset.py \
    --dataset_path /path/to/dataset \
    --output_path /path/to/output \
    --ratio 0.8 \
    --seed 42 \
    --is_local
```

## SLURM Job Scripts
SLURM job scripts are provided for executing the dataset combination and sampling scripts on a computing cluster.
- `combine_dataset.sh`  This script submits a job to combine two datasets.
- `sample_dataset.sh` This script submits a job to sample a dataset.
