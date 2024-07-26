from argparse import ArgumentParser

from continue_pretraining.tokenizer.huggingface.trainer import train

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="The path and prefix to use when saving the trained tokenizer.",
    )
    parser.add_argument(
        "--load_dataset_path",
        type=str,
        required=True,
        help="The name or path of the Hugging Face dataset to load.",
    )
    parser.add_argument(
        "--load_dataset_name",
        type=str,
        default=None,
        help="The name of the dataset split to use. Defaults to None.",
    )
    parser.add_argument(
        "--is_local",
        default=False,
        help="Whether the dataset is local directory. Defaults to False.",
    )
    parser.add_argument(
        "--batch_size",
        type=str,
        default=1000,
        help="The size of the batch to use when training the tokenizer. Defaults to 1000.",  # noqa: E501
    )
    parser.add_argument(
        "--vocab_size",
        type=str,
        default=32000,
        help="The size of the vocabulary to use when training the tokenizer. Defaults to 32000.",  # noqa: E501
    )

    args = parser.parse_args()

    train(
        output_path=args.output_path,
        load_dataset_path=args.load_dataset_path,
        load_dataset_name=args.load_dataset_name,
        is_local=args.is_local,
        batch_size=args.batch_size,
        vocab_size=args.vocab_size,
    )
