import os
from typing import Optional
import sentencepiece as spm
from transformers import LlamaTokenizer
from datasets import Dataset, load_dataset, load_from_disk

from continue_pretraining.tokenizer.sentencepiece.constants import (
    TEXT_COLUMN,
    EOS_TOKEN,
    BOS_TOKEN,
    UNK_TOKEN,
    UNIGRAM_MODE,
    WORD_MODE,
    CHAR_MODE,
    BPE_MODE,
    TRAIN_SPLIT,
)
from tqdm import tqdm


def dataset_iterator(dataset: Dataset, text_column: str):
    """
    A generator function that iterates over a dataset and yields the text data from the specified column.

    Args:
        dataset (Dataset): The dataset to iterate over.
        text_column (str): The name of the column containing the text data.

    Yields:
        str: The text data from the specified column for each row in the dataset.
    """  # noqa: E501
    # Iterate over the dataset using tqdm to show a progress bar
    for i in tqdm(range(len(dataset))):
        # Yield the text data from the specified column
        yield dataset[i][text_column]


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
    else:
        # Load dataset from local disk
        text_dataset = load_from_disk(load_dataset_path)[TRAIN_SPLIT]

    # Ensure the output directory exists
    os.makedirs(output_path, exist_ok=True)

    # Train SentencePiece tokenizer
    spm.SentencePieceTrainer.train(
        sentence_iterator=iter(dataset_iterator(text_dataset, TEXT_COLUMN)),
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
