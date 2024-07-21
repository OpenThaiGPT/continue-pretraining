from transformers import AutoTokenizer
from datasets import load_dataset, load_from_disk
import argparse
import os

from continue_pretraining.model.preprocess import tokenize_function

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--tokenizer_name_or_path",
        required=True,
        type=str,
        help="The name or path of the tokenizer to use.",
    )
    parser.add_argument(
        "--output_path",
        required=True,
        type=str,
        help="The path where the processed dataset will be saved.",
    )
    parser.add_argument(
        "--dataset_path",
        required=True,
        type=str,
        help="The path to the dataset to load.",
    )
    parser.add_argument(
        "--dataset_name",
        default=None,
        type=str,
        help="The name of the dataset (if loading from a dataset repository).",
    )
    parser.add_argument(
        "--is_local",
        action="store_true",
        default=False,
        help="Flag indicating whether the dataset is a local directory. Defaults to False.",  # noqa: E501
    )
    parser.add_argument(
        "--max_sequence_length",
        default=2048,
        type=int,
        help="The maximum sequence length for tokenized sequences. Defaults to 2048.",  # noqa: E501
    )
    parser.add_argument(
        "--num_proc",
        default=os.cpu_count(),
        type=int,
        help="The number of processes to use for tokenization. Defaults to the number of CPU cores.",  # noqa: E501
    )

    # Parse the command-line arguments
    args = parser.parse_args()

    # Load the tokenizer from the specified name or path
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)

    # Load the dataset from the specified path
    if not args.is_local:
        # Load dataset from a dataset repository
        dataset = load_dataset(
            path=args.dataset_path,
            name=args.dataset_name,
            trust_remote_code=True,
        )
    else:
        # Load dataset from a local directory
        dataset = load_from_disk(args.dataset_path)

    # Apply the tokenization function to the dataset
    dataset = dataset.map(
        tokenize_function(
            tokenizer,
            max_sequence_length=args.max_sequence_length,
        ),
        batched=True,
        remove_columns=dataset["train"].column_names,
        num_proc=args.num_proc,
    )

    # Save the processed dataset to the specified output path
    dataset.save_to_disk(args.output_path)
