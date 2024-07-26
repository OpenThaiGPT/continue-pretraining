from tokenizers import ByteLevelBPETokenizer
from datasets import load_from_disk, load_dataset, Dataset
from typing import Optional
from tqdm import tqdm
import os

from continue_pretraining.tokenizer.huggingface.constants import (
    TRAIN_SPLIT,
    TEXT_COLUMN,
    BOS_TOKEN,
    EOS_TOKEN,
    UNK_TOKEN,
)


def dataset_batch_iterator(
    dataset: Dataset,
    text_column: str,
    batch_size: int = 1000,
):
    """
    A generator function that iterates over a dataset in batches and yields lists of text data from the specified column.

    Args:
        dataset (Dataset): The dataset to iterate over.
        text_column (str): The name of the column containing the text data.
        batch_size (int, optional): The number of rows to include in each batch. Defaults to 1000.

    Yields:
        List[str]: A list of text data from the specified column for each batch in the dataset.
    """  # noqa: E501
    # Iterate over the dataset in steps of batch_size using tqdm to show a progress bar  # noqa: E501
    for i in tqdm(range(0, len(dataset), batch_size)):
        # Yield a list of text data from the specified column for the current batch  # noqa: E501
        yield [
            text for text in dataset[i : i + batch_size][text_column]  # noqa: E203 E501
        ]


def train(
    output_path: str,
    load_dataset_path: str,
    load_dataset_name: Optional[str] = None,
    is_local: bool = False,
    batch_size: int = 1000,
    vocab_size: int = 32000,
) -> None:
    """
    Train a ByteLevelBPETokenizer on a large text dataset.

    Args:
        output_path (str): The path to use when saving the trained tokenizer.
        load_dataset_path (str): The name or path of the Hugging Face dataset to load.
        load_dataset_name (Optional[str]): The name of the dataset split to use. Defaults to None.
        is_local (bool): Whether the dataset is in a local directory. Defaults to False.
        batch_size (int): The size of the batch to use when training the tokenizer. Defaults to 1000.
        vocab_size (int): The size of the vocabulary to use when training the tokenizer. Defaults to 32000.

    Returns:
        None
    """  # noqa: E501

    # Load the dataset
    if not is_local:
        dataset = load_dataset(
            path=load_dataset_path,
            name=load_dataset_name,
            split=TRAIN_SPLIT,
            trust_remote_code=True,
        )
    else:
        # Load dataset from local disk
        dataset = load_from_disk(load_dataset_path)[TRAIN_SPLIT]

    # Instantiate the ByteLevelBPETokenizer
    tokenizer = ByteLevelBPETokenizer()

    # Train the tokenizer on the dataset
    tokenizer.train_from_iterator(
        dataset_batch_iterator(dataset, TEXT_COLUMN, batch_size),
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=[BOS_TOKEN, EOS_TOKEN, UNK_TOKEN],
    )

    # Save the trained tokenizer to disk
    os.makedirs(output_path, exist_ok=True)
    tokenizer.save(output_path + "/tokenizer.json")
    tokenizer.save_model(output_path)
