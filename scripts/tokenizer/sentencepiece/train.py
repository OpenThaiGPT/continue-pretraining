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
        "--num_threads",
        type=int,
        default=os.cpu_count(),
        help="The number of threads to use when training the tokenizer. Defaults to the number of available CPU cores.",  # noqa: E501
    )
    parser.add_argument(
        "--load_dataset_path",
        type=str,
        default="oscar",
        help='The name or path of the Hugging Face dataset to load. Defaults to "oscar".',  # noqa: E501
    )
    parser.add_argument(
        "--load_dataset_name",
        type=str,
        default=None,
        help="The name of the dataset split to use. Defaults to None.",  # noqa: E501
    )
    parser.add_argument(
        "--is_local",
        action="store_true",
        default=False,
        help="Whether the dataset is local directory. Defaults to False.",
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
        num_threads=args.num_threads,
        load_dataset_path=args.load_dataset_path,
        load_dataset_name=args.load_dataset_name,
        is_local=args.is_local,
        large_corpus=args.large_corpus,
        mode=args.mode,
    )
