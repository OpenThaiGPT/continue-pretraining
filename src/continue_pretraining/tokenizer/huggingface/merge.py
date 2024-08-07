from transformers import PreTrainedTokenizerFast, GPT2TokenizerFast
import json
import os
from typing import Dict

from continue_pretraining.tokenizer.huggingface.constants import (
    TEMP_FOLDER,
    UTF8_ENCODING,
    MERGE_VOCAB_FILE,
    NEW_MERGED_RULE,
)


def merge(
    main_tokenizer_path: str,
    add_tokenizer_path: str,
    base_merge_file: str,
    add_merge_file: str,
):
    """
    Merges two tokenizers into one by combining their vocabularies and merge files.

    Args:
        main_tokenizer_path (str): Directory of the base tokenizer.
        add_tokenizer_path (str): Directory of the additional tokenizer.
        base_merge_file (str): File path of the base tokenizer's merge rules.
        add_merge_file (str): File path of the additional tokenizer's merge rules.

    Returns:
        GPT2TokenizerFast: The merged tokenizer.
    """  # noqa: E501
    # Load the base and additional tokenizers
    base_tokenizer = PreTrainedTokenizerFast.from_pretrained(
        main_tokenizer_path,
    )
    add_tokenizer = PreTrainedTokenizerFast.from_pretrained(
        add_tokenizer_path,
    )

    # Retrieve the vocabularies from both tokenizers
    base_vocab = base_tokenizer.get_vocab()
    add_vocab = add_tokenizer.get_vocab()

    # Create a folder to store the new merged vocabulary and merge file
    os.makedirs(TEMP_FOLDER, exist_ok=True)

    # Create a new vocabulary by merging the base and additional vocabularies
    new_vocab: Dict[str, int] = {}
    idx = 0
    for word in base_vocab.keys():
        if word not in new_vocab.keys():
            new_vocab[word] = base_vocab[word]
            idx += 1

    # Add words from the additional tokenizer if they are not already in the new vocabulary  # noqa: E501
    for word in add_vocab.keys():
        if word not in new_vocab.keys():
            new_vocab[word] = idx
            idx += 1

    # Convert the new vocabulary dictionary to a JSON string
    new_vocab_json = json.dumps(new_vocab, ensure_ascii=False)

    # Write the new vocabulary JSON to a file
    vocab_file_path = os.path.join(TEMP_FOLDER, MERGE_VOCAB_FILE)
    with open(vocab_file_path, "w", encoding=UTF8_ENCODING) as outfile:
        outfile.write(new_vocab_json)

    # Merge the base and additional merge files
    merge_file_path = os.path.join(TEMP_FOLDER, NEW_MERGED_RULE)
    with open(base_merge_file, "r", encoding=UTF8_ENCODING) as f1, open(
        add_merge_file, "r", encoding=UTF8_ENCODING
    ) as f2, open(merge_file_path, "w", encoding=UTF8_ENCODING) as out_file:
        # Ignore the first line of each input file
        next(f1)
        next(f2)

        # Read the remaining lines of each file and write unique lines to the output file  # noqa: E501
        lines = set()
        for line in f1:
            if line not in lines:
                out_file.write(line)
                lines.add(line)
        for line in f2:
            if line not in lines:
                out_file.write(line)
                lines.add(line)

    # Create the merged tokenizer using the new vocabulary and merge file
    merge_tokenizer = GPT2TokenizerFast(
        vocab_file=vocab_file_path,
        merges_file=merge_file_path,
        unk_token=base_tokenizer.unk_token,
        eos_token=base_tokenizer.eos_token,
        bos_token=base_tokenizer.bos_token,
    )

    return merge_tokenizer
