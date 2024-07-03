import os
from typing import Optional
import sentencepiece as spm
from transformers import LlamaTokenizer
from datasets import load_dataset, load_from_disk

from continue_pretraining.tokenizer.sentencepiece.constants import (
    PREPARE_DATASETS_KEY,
    DOC_TEXT,
    EOS_TOKEN,
    BOS_TOKEN,
    UNK_TOKEN,
    UNIGRAM_MODE,
    WORD_MODE,
    CHAR_MODE,
    BPE_MODE,
    TRAIN_SPLIT,
)


class DataSetColumnIterator:
    """
    An iterator class for iterating over a specific column in a dataset.

    Attributes:
        dataset (iterable): The dataset to iterate over.
        column_name (str): The name of the column to extract from each item in the dataset.

    Methods:
        __iter__(): Returns an iterator over the specified column values.
    """  # noqa: E501

    def __init__(self, dataset, column_name: str):
        """
        Initializes the DataSetColumnIterator with the dataset and the column name.

        Args:
            dataset (iterable): The dataset to iterate over.
            column_name (str): The name of the column to extract from each item in the dataset.
        """  # noqa: E501
        self.dataset = iter(dataset)
        self.column_name = column_name

    def __iter__(self):
        """
        Iterates over the dataset, yielding values from the specified column.

        Yields:
            The value from the specified column in each item of the dataset.

        Raises:
            ValueError: If the specified column name is not found in an item of the dataset.
        """  # noqa: E501
        for item in self.dataset:
            try:
                yield item[self.column_name]
            except KeyError:
                raise ValueError(
                    f"Column '{self.column_name}' is not a valid index for the dataset"  # noqa: E501
                )


def train_tokenizer(
    output_path: str,
    vocab_size: int,
    num_threads: Optional[int] = os.cpu_count(),
    load_dataset_path: str = "oscar",
    load_dataset_name: Optional[str] = None,
    is_local: bool = False,
    large_corpus: bool = False,
    mode: str = BPE_MODE,
) -> None:
    """
    Train a SentencePiece tokenizer on a large text dataset.

    Args:
        output_path (str): The path and prefix to use when saving the trained tokenizer.
        vocab_size (int): The size of the vocabulary to use when training the tokenizer.
        num_threads (int, optional): The number of threads to use when training the tokenizer. Defaults to the number of available CPU cores.
        load_dataset_path (str, optional): The name or path of the Hugging Face dataset to load. Defaults to "oscar".
        load_dataset_name (str, optional): The name of the dataset split to use. Defaults to None.
        is_local (bool): Whether the dataset is in a local directory. Defaults to False.
        large_corpus (bool): Whether the code is running on a large dataset. Defaults to False.
        mode (str): The training model of the tokenizer. Defaults to 'bpe'.

    Returns:
        None
    """  # noqa: E501
    # Validate mode argument
    if mode not in {UNIGRAM_MODE, BPE_MODE, WORD_MODE, CHAR_MODE}:
        raise KeyError(
            f"mode must be one of {UNIGRAM_MODE}, {WORD_MODE}, {CHAR_MODE}, or {BPE_MODE}"  # noqa: E501
        )

    # Load dataset from Hugging Face if not local
    if not is_local:
        text_dataset = load_dataset(
            path=load_dataset_path,
            name=load_dataset_name,
            split=TRAIN_SPLIT,
            trust_remote_code=True,
        )
        text_dataset = text_dataset.to_iterable_dataset()
    else:
        # Load dataset from local disk
        text_dataset = load_from_disk(load_dataset_path)

    # Process dataset
    text_processed_dataset = text_dataset.map(
        function=lambda x: {PREPARE_DATASETS_KEY: [t for t in x[DOC_TEXT]]},
        batched=True,
    )

    # Ensure the output directory exists
    os.makedirs(output_path, exist_ok=True)

    # Train SentencePiece tokenizer
    spm.SentencePieceTrainer.train(
        sentence_iterator=iter(
            DataSetColumnIterator(text_processed_dataset, PREPARE_DATASETS_KEY)
        ),
        model_prefix=output_path + "/spm_tokenizer",
        vocab_size=vocab_size,
        user_defined_symbols=[],
        num_threads=num_threads,
        train_extremely_large_corpus=large_corpus,
        model_type=mode,
    )

    # Load and configure the tokenizer
    tokenizer = LlamaTokenizer(vocab_file=output_path + "/spm_tokenizer.model")
    tokenizer.eos_token = EOS_TOKEN
    tokenizer.bos_token = BOS_TOKEN
    tokenizer.unk_token = UNK_TOKEN

    # Save the tokenizer
    tokenizer.save_pretrained(output_path)
