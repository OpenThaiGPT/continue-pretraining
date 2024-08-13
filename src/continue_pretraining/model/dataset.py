from datasets import load_from_disk
from typing import Dict
import os

from continue_pretraining.model.data_collator import (
    DataCollatorForPretraining,
)
from continue_pretraining.model.args import DataArguments


def make_data_module(
    data_args: DataArguments,
) -> Dict:
    """
    Creates a data module for pretraining, including datasets and data collator.

    Args:
        data_args (DataArguments): Configuration for loading datasets.

    Returns:
        Dict: A dictionary containing the training dataset, evaluation dataset, and data collator.
    """  # noqa: E501
    train_dataset = load_from_disk(
        os.path.join(data_args.data_path, data_args.train_split)
    )
    eval_dataset = load_from_disk(
        os.path.join(data_args.data_path, data_args.eval_split)
    )
    data_collator = DataCollatorForPretraining()
    return {
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": data_collator,
    }
