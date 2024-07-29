from typing import Sequence, Dict
import torch


class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised pre-training."""

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [instance["input_ids"] for instance in instances]
        input_ids = torch.tensor(input_ids)  # type: ignore
        return {
            "input_ids": input_ids,  # type: ignore
            "labels": input_ids,  # type: ignore
        }
