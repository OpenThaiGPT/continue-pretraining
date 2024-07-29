from datasets import load_from_disk
from typing import Dict, List, Optional
import os

from continue_pretraining.model.dataset.combine_dataset import CombinedDataset
from continue_pretraining.model.dataset.data_collator import (
    DataCollatorForSupervisedDataset,
)
from continue_pretraining.model.args import DataArguments


def load_dataset(
    paths: List[str],
    weights: Optional[List[float]] = None,
    split: str = "train",
    seed: int = 42,
):
    """
    Loads a dataset from disk and applies weights to the data files.

    Args:
        paths (List[str]): List of paths to the data files.
        weights (Optional[List[float]]): Weights for each dataset file by reference on minimun dataset. If None, equal weights are assigned..
        split (str): Name of the split to load (e.g., 'train' or 'eval'). Defaults to 'train'.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.

    Returns:
        CombinedDataset: A combined dataset with the specified weights.
    """  # noqa: E501
    datasets = []
    for path in paths:
        path_to_split = os.path.join(path, split)
        dataset = load_from_disk(path_to_split)
        datasets.append(dataset)
    return CombinedDataset(datasets, seed, weights)


def make_supervised_data_module(
    data_args: DataArguments,
    seed: int = 42,
) -> Dict:
    """
    Creates a data module for pretraining, including datasets and data collator.

    Args:
        data_args (DataArguments): Configuration for loading datasets.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.

    Returns:
        Dict: A dictionary containing the training dataset, evaluation dataset, and data collator.
    """  # noqa: E501
    train_dataset = load_dataset(
        data_args.data_path,
        data_args.data_weights,
        data_args.train_split,
        seed,
    )
    eval_dataset = load_dataset(
        data_args.data_path,
        data_args.data_weights,
        data_args.eval_split,
        seed,
    )
    data_collator = DataCollatorForSupervisedDataset()
    return {
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": data_collator,
    }
