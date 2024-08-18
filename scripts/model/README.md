# Model Training
Contains scripts to tokenize datasets, model vocabulary expansion and continue pretraining. The processes are designed to be run on a computing cluster using SLURM.

## Usage

### Tokenize Datasets
The `preprocessing/preprocess_dataset.py` script tokenizes a dataset and pads sequences to a specified maximum length. 

#### Command-line Arguments
- `--tokenizer_name_or_path`: Name or path of the tokenizer to use (required).
- `--output_path`: Path to save the processed dataset (required).
- `--dataset_path`: Path to the dataset to load (required).
- `--dataset_name`: Name of the dataset if loading from a dataset repository (optional).
- `--is_local`: Flag indicating if the dataset is loaded from a local directory (optional, default: False).
- `--max_sequence_length`: Maximum sequence length for tokenized sequences (optional, default: 2048).
- `--num_proc`: Number of processes for tokenization (optional, default: number of CPU cores).

#### Example Usage
```bash
python preprocessing/preprocess_dataset.py \
    --tokenizer_name_or_path /path/to/tokenizer \
    --dataset_path /path/to/dataset_1 \
    --output_path /path/to/output \
    --max_sequence_length 2048 \
    --is_local
```

### Model Vocabulary Expansion
The `prepare_model/prepare_model.py` script updates the model's token embedding size to match the tokenizer's vocabulary size and saves the updated model and tokenizer.

#### Command-line Arguments
- `--model_name_or_path`: Path or Hugging Face repository to load the model from (required).
- `--tokenizer_path`: Path to the tokenizer (required).
- `--output_path`: Path to save the updated model and tokenizer (required).

#### Example Usage
```bash
python prepare_model/prepare_model.py \
    --model_name_or_path /path/to/model \
    --tokenizer_path /path/to/tokenizer \
    --output_path /path/to/output \
```

### Continue Pretraining
The `training/train.py` script sets up a Hugging Face Trainer to pre-train a model for causal language modeling using distributed resources.

#### Command-line Arguments
- `--model_name_or_path`: Path or Hugging Face repository of the pre-trained model (required).
- `--tokenizer_name_or_path`: Path to the tokenizer (required).
- `--data_path`: Path to the tokenized dataset (required).
- `--train_split`: Name of the training dataset split (optional, default: "train").
- `--eval_split`: Name of the evaluation dataset split (optional, default: "eval").
- `--cache_dir`: Path to store the cache pre-trained models downloaded from [huggingface.co](https://huggingface.co/) (optional).
- `--optim`: Name of the Huggingface optimizer to use (optional, default: "adamw_torch").
- `--checkpoint`: Path to a specific checkpoint to resume training (optional).
> Note: You can use arguments from [huggingface training arguments](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments)

#### Example Usage
```bash
python training/train.py \
    --model_name_or_path /path/to/model \
    --tokenizer_name_or_path /path/to/tokenizer \
    --data_path /path/to/data \
    --data_seed 42 \
    --train_split train \
    --eval_split eval \
    --bf16 True \
    --output_dir /path/to/output \
    --num_train_epochs 3 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --tf32 True \
    --gradient_checkpointing True \
```


## SLURM Job Scripts
SLURM job scripts are provided to run the tokenization, model resizing and model continue pretraining script on a computing cluster.
- `preprocessing/submit_preprocess_dataset.sh` Submits a job for dataset tokenization.
- `prepare_model/submit_prepare_model.sh` Submits a job to resize model embeddings and save the model and tokenizer.
-  `training/submit_multinode.sh` Submits a job to multi-node job to pre-training a language model using distributed training with accelerate.

