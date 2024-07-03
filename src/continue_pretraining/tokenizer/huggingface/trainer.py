from tokenizers import ByteLevelBPETokenizer
from datasets import load_from_disk, load_dataset
from nlpo3 import segment, load_dict
from typing import Optional
from tqdm import tqdm
import os

from continue_pretraining.tokenizer.huggingface.constants import (
    DICT_NAME,
    TRAIN_SPLIT,
    TEXT_COLUMN,
    BOS_TOKEN,
    EOS_TOKEN,
    UNK_TOKEN,
)


def train(
    output_path: str,
    dict_file: str,
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
        dict_file (str): The path to the vocabulary file for newmm tokenization.
                         The file can be obtained from https://github.com/PyThaiNLP/pythainlp/blob/dev/pythainlp/corpus/words_th.txt.
        load_dataset_path (str): The name or path of the Hugging Face dataset to load.
        load_dataset_name (Optional[str]): The name of the dataset split to use. Defaults to None.
        is_local (bool): Whether the dataset is in a local directory. Defaults to False.
        batch_size (int): The size of the batch to use when training the tokenizer. Defaults to 1000.
        vocab_size (int): The size of the vocabulary to use when training the tokenizer. Defaults to 32000.

    Returns:
        None
    """  # noqa: E501

    # Load the dictionary for tokenization
    load_dict(dict_file, DICT_NAME)

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

    def th_tokenize(text):
        """
        Tokenize text using the Thai newmm tokenizer with the loaded dictionary.

        Args:
            text (str): The text to tokenize.

        Returns:
            str: The tokenized text.
        """  # noqa: E501
        result = " ".join(segment(text, DICT_NAME))
        return result

    def batch_iterator(batch_size=1000):
        """
        Iterator to yield batches of tokenized text from the dataset.

        Args:
            batch_size (int): The size of the batches to yield.

        Yields:
            List[str]: A batch of tokenized text.
        """
        for i in tqdm(range(0, len(dataset), batch_size)):
            yield [
                th_tokenize(text)
                for text in dataset[i : i + batch_size][TEXT_COLUMN]  # noqa: E203 E501
            ]

    # Train the tokenizer on the dataset
    tokenizer.train_from_iterator(
        batch_iterator(batch_size),
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=[BOS_TOKEN, EOS_TOKEN, UNK_TOKEN],
    )

    # Save the trained tokenizer to disk
    os.makedirs(output_path, exist_ok=True)
    tokenizer.save(output_path + "/tokenizer.json")
    tokenizer.save_model(output_path)
