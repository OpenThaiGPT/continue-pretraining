from typing import Sequence, Dict
import torch


class DataCollatorForPretraining(object):
    """
    Collate examples for pretraining.

    ---

    จัดเก็บข้อมูลตัวอย่างสำหรับการ pretraining
    """

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [instance["input_ids"] for instance in instances]
        input_ids = torch.tensor(input_ids)  # type: ignore
        return {
            "input_ids": input_ids,  # type: ignore
            "labels": input_ids,  # type: ignore
        }
