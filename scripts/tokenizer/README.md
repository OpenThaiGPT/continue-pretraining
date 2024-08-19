# Tokenizer Management
Contains scripts to train and merge tokenizers using the HuggingFace or SentencePiece tokenizer libraries. The processes are designed to be run on a computing cluster using SLURM.

## Usage

### Huggingface Tokenizer
- ### Training 
    The `huggingface/train.py` script trains a BPE tokenizer using HuggingFace tokenizers.

    #### Command-line Arguments
    - `--output_path`: Path to save the tokenizer (required).
    - `--load_dataset_path`: Name or path of the Hugging Face dataset to load (required).
    - `--load_dataset_name`: Name of the dataset split to use (optional).
    - `--is_local`: Flag indicating if the dataset is loaded from a local directory (optional, default: False).
    - `--batch_size`: Batch size to use when training the tokenizer (optional, default: 1000).
    - `--vocab_size`: Size of the vocabulary to use when training the tokenizer (optional, default: 32000).

    #### Example Usage
    ```bash
    python huggingface/train.py \
        --output_path /path/to/output \
        --load_dataset_path /path/to/dataset \
        --is_local \
        --batch_size 1000 \
        --vocab_size 32000 \
    ```

- ### Merge
    The `huggingface/merge.py` script combines two BPE Huggingface tokenizers.

    #### Command-line Arguments
    - `--main_tokenizer_path`: Path of the main tokenizer (required).
    - `--add_tokenizer_path`: Path of the additional tokenizer (required).
    - `--base_merge_file`: Path of the base tokenizer's merge rules (optional).
    - `--add_merge_file`: Path of the additional tokenizer's merge rules (optional).
    - `--output_path`: Path to save the merged tokenizer (required).

    > Note: if don't assign `--base_merge_file` / `--add_merge_file` will use merge file path from `--main_tokenizer_path` / `--add_merge_file`

    #### Example Usage
    ```bash
    python huggingface/merge.py \
        --base_tokenizer_dir /path/to/original/hugginface/tokenizer \
        --add_tokenizer_dir /path/to/extra/hugginface/tokenizer \
        --output_dir /path/to/output \
    ```

### SentencePiece Tokenizer
- ### Training
    The `sentencepiece/train.py` script training Sentencepiece tokenizer.

    #### Command-line Arguments
    - `--output_path`: Path and prefix to use when saving the trained tokenizer (required).
    - `--vocab_size`: Size of the vocabulary to use when training the tokenizer (optional, default: 32,000).
    - `--num_threads`: Number of threads to use when training the tokenizer (optional, default: available CPU cores).
    - `--load_dataset_path`: Name or path of the Hugging Face dataset to load (optional, default: "oscar").
    - `--load_dataset_name`: Name of the dataset split to use.
    - `--is_local`: Flag to indicate if the dataset is a local directory (optional, default: False).
    - `--large_corpus`: Flag to indicate if the dataset is large (optional, default: False).
    - `--mode`: Tokenizer training mode to use. Options: `unigram`, `bpe`, `char`, `word` (optional, default: `unigram`).

    #### Example Usage
    ```bash
    python sentencepiece/train.py \
        --output_path ./path/to/output \
        --vocab_size 32000 \
        --load_dataset_path /path/to/dataset \
        --mode bpe \
        --large_corpus \
        --is_local
    ```

- ### Merge
    The `sentencepiece/merge.py` script combines between Sentencepiece and Huggingface Llama tokenizer.

    #### Command-line Arguments
    - `--main_tokenizer_path`: Path to original llama tokenizer (required).
    - `--add_tokenizer_path`: Path to tokenizer for adding vocab (required).
    - `--output_path`: Path of output tokenizer (optional).

    #### Example Usage
    ```bash
    python sentencepiece/merge.py \
        --main_tokenizer_path /path/to/original/llama/tokenizer \
        --main_tokenizer_path /path/to/extra/sentencepiece/tokenizer \
        --output_path /path/to/save \
    ```

## SLURM Job Scripts
SLURM job scripts are provided to run the tokenizer training and merging scripts on a computing cluster.

- `huggingface/submit_train.sh` Submits a job for training a HuggingFace tokenizer.
- `huggingface/submit_merge.sh` Submits a job to merge two HuggingFace tokenizers.
- `sentencepiece/submit_train.sh` Submits a job for training a SentencePiece tokenizer.
- `sentencepiece/submit_merge.sh` Submits a job to merge a SentencePiece and a HuggingFace tokenizer.
