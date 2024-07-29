from datasets import IterableDataset
from typing import List, Optional
import random


class CombinedDataset(IterableDataset):
    """
    A combined dataset class that allows for the weighted combination of multiple datasets.

    Args:
        datasets (List[IterableDataset]): List of datasets to be combined.
        seed (int): Random seed for reproducibility. Defaults to 42.
        weights (Optional[List[float]]): Weights for each dataset by reference on minimun dataset. If None, equal weights are assigned. Defaults to None.
    """  # noqa: E501

    def __init__(
        self,
        datasets: List[IterableDataset],
        seed: int = 42,
        weights: Optional[List[float]] = None,
    ):
        self._seed = seed
        self._datasets = datasets
        self._weights = weights

        n_datasets = len(datasets)

        if weights is None:
            self._weights = [1 / n_datasets] * n_datasets

        len_datasets = []
        for dataset in self._datasets:
            len_datasets.append(len(dataset))
        self.total_len = int(min(len_datasets) * sum(self._weights))

    def __iter__(self):
        """
        Returns an iterator for the combined dataset.
        """
        return CombinedDatasetIterator(
            self._datasets,
            self._weights,
            self._seed,
        )

    def __len__(self):
        """
        Returns the total length of the combined dataset.
        """
        return self.total_len


class CombinedDatasetIterator:
    """
    Iterator for the CombinedDataset class.

    Args:
        datasets (List[IterableDataset]): List of datasets to be iterated over.
        weights (List[float]): Weights for each dataset.
        seed (int): Random seed for reproducibility. Defaults to 42.
    """

    def __init__(
        self,
        datasets: List[IterableDataset],
        weights: List[float],
        seed: int = 42,
    ):
        self._datasets = [iter(el) for el in datasets]
        self._weights = weights
        self._rng = random.Random(seed)

    def __next__(self):
        """
        Returns the next element from the combined dataset based on the assigned weights.
        """  # noqa: E501
        (dataset,) = self._rng.choices(
            self._datasets,
            weights=self._weights,
            k=1,
        )

        return next(dataset)
