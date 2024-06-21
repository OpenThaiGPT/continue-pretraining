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
    Args:
        output_path (str): The path and prefix to use when saving the trained tokenizer.
        dict_file (str): The path and prefix to vocabulary for newmm tokenization. You can get from https://github.com/PyThaiNLP/pythainlp/blob/dev/pythainlp/corpus/words_th.txt.
        load_dataset_path (str): The name or path of the Hugging Face dataset to load.
        load_dataset_name (optional, str): The name of the dataset split to use. Defaults to None.
        is_local (bool): Whether the dataset is local directory. Defaults to False.
        batch_size (int): The size of the batch to use when training the tokenizer. Defaults to 1000.
        vocab_size (int): The size of the vocabulary to use when training the tokenizer. Defaults to 32000.
    Returns:
        None
    """  # noqa: E501
    load_dict(dict_file, DICT_NAME)

    # load dataset
    if not is_local:
        dataset = load_dataset(
            path=load_dataset_path,
            name=load_dataset_name,
            split=TRAIN_SPLIT,
            trust_remote_code=True,
        )
    else:
        # load dataset from local dataset
        dataset = load_from_disk(load_dataset_path)[TRAIN_SPLIT]

    # Instantiate tokenizer
    tokenizer = ByteLevelBPETokenizer()

    def th_tokenize(text):
        result = " ".join(segment(text, DICT_NAME))
        return result

    def batch_iterator(batch_size=1000):
        for i in tqdm(range(0, len(dataset), batch_size)):
            yield [
                th_tokenize(text)
                for text in dataset[i : i + batch_size][TEXT_COLUMN]  # noqa
            ]

    # Customized training
    tokenizer.train_from_iterator(
        batch_iterator(batch_size),
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=[BOS_TOKEN, EOS_TOKEN, UNK_TOKEN],
    )

    # Save files to disk
    os.makedirs(output_path, exist_ok=True)
    tokenizer.save(output_path + "/tokenizer.json")
    tokenizer.save_model(output_path)
