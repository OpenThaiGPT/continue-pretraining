import os
from typing import Optional
import sentencepiece as spm
from transformers import LlamaTokenizer
from datasets import load_dataset, load_from_disk

from .constants import (
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


def prepare_datasets(texts: dict) -> dict:
    """
    Preprocesses a list of text documents and returns a dictionary with a single key 'PREPARE_DATASETS_KEY'
    that maps to the preprocessed texts.

    Args:
        texts (dict): A dictionary containing a key 'DOC_TEXT' that maps to a list of text documents.

    Returns:
        dict: A dictionary with a single key 'PREPARE_DATASETS_KEY' that maps to a list of preprocessed text documents.
    """  # noqa: E501
    preapared_texts = []
    for text in texts[DOC_TEXT]:  # for every doc
        # write custom preprocessing
        preapared_texts.append(text)

    return {PREPARE_DATASETS_KEY: preapared_texts}


class DataSetColumnIterator:
    def __init__(self, dataset, column_name: str):
        self.dataset = iter(dataset)
        self.column_name = column_name

    def __iter__(self):
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
        load_dataset_name (str, optional): The name of the dataset split to use. Defaults to "unshuffled_deduplicated_th".
        is_local (bool): Whether the dataset is local directory. Defaults to False.
        large_corpus (bool): Whether the code is running on large dataset. Defaults to False.
        mode (bool): The training model of tokenizer. Defaults to unigram.

    Returns:
        None
    """  # noqa: E501

    if not (
        mode == UNIGRAM_MODE
        or mode == BPE_MODE
        or mode == WORD_MODE
        or mode == CHAR_MODE
    ):
        KeyError(
            f"mode mush be {UNIGRAM_MODE} {WORD_MODE} {CHAR_MODE} or {BPE_MODE}"  # noqa: E501
        )

    if not is_local:
        text_dataset = load_dataset(
            path=load_dataset_path,
            name=load_dataset_name,
            split=TRAIN_SPLIT,
            trust_remote_code=True,
        )

        text_dataset = text_dataset.to_iterable_dataset()

    else:
        # Stream from local files
        text_dataset = load_from_disk(load_dataset_path)

    text_processed_dataset = text_dataset.map(
        function=prepare_datasets,
        batched=True,
    )

    os.makedirs(output_path, exist_ok=True)

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

    tokenizer = LlamaTokenizer(vocab_file=output_path + "/spm_tokenizer.model")

    tokenizer.eos_token = EOS_TOKEN
    tokenizer.bos_token = BOS_TOKEN
    tokenizer.unk_token = UNK_TOKEN

    tokenizer.save_pretrained(output_path)
