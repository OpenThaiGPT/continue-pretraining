from argparse import ArgumentParser
import os

from continue_pretraining.tokenizer.sentencepiece import (
    trainer,
    constants,
)

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="The path and prefix to use when saving the trained tokenizer.",  # noqa: E501
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=32000,
        help="The size of the vocabulary to use when training the tokenizer.",
    )
    parser.add_argument(
        "--num_docs",
        type=int,
        default=None,
        help="The number of documents to use from the input dataset.",
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=os.cpu_count(),
        help="The number of CPU cores to use when training the tokenizer. Defaults to the number of available CPU cores.",  # noqa: E501
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        default=False,
        help="Whether the code is running on a Slurm cluster. Defaults to False.",  # noqa: E501
    )
    parser.add_argument(
        "--load_dataset_path",
        type=str,
        default="oscar",
        help='The name of the Hugging Face dataset to load. Defaults to "oscar".',  # noqa: E501
    )
    parser.add_argument(
        "--load_dataset_name",
        type=str,
        default="unshuffled_deduplicated_th",
        help='The name of the dataset split to use. Defaults to "unshuffled_deduplicated_th".',  # noqa: E501
    )
    parser.add_argument(
        "--load_dataset_local_path",
        type=str,
        default=None,
        help="The path to a local directory containing the input data. If specified, the Hugging Face dataset is not used. Defaults to None.",  # noqa: E501
    )
    parser.add_argument(
        "--load_dataset_data_type",
        type=str,
        default=None,
        help='The file type of the input data if using a local directory. Defaults to "csv".',  # noqa: E501
    )
    parser.add_argument(
        "--large_corpus",
        action="store_true",
        default=False,
        help="Whether the code is running on large dataset. Defaults to False.",  # noqa: E501
    )
    parser.add_argument(
        "--mode",
        default=constants.UNIGRAM_MODE,
        choices=[
            constants.BPE_MODE,
            constants.CHAR_MODE,
            constants.WORD_MODE,
            constants.UNIGRAM_MODE,
        ],
        help="The training model of tokenizer. Defaults to unigram.",
    )

    args = parser.parse_args()

    trainer.train_tokenizer(
        output_path=args.output_path,
        vocab_size=args.vocab_size,
        num_docs=args.num_docs,
        num_proc=args.num_proc,
        streaming=args.streaming,
        load_dataset_path=args.load_dataset_path,
        load_dataset_name=args.load_dataset_name,
        load_dataset_local_path=args.load_dataset_local_path,
        load_dataset_data_type=args.load_dataset_data_type,
        large_corpus=args.large_corpus,
        mode=args.mode,
    )
