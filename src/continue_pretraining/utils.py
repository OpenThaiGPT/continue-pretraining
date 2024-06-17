from datasets import load_from_disk, load_dataset
import os


def load_local_dataset(data_type, local_path):
    if data_type is None:
        text_dataset = load_from_disk(local_path)["train"]
    else:
        data_files = {
            "train": [
                f"{local_path}/{filename}"
                for filename in os.listdir(local_path)  # noqa: E501
            ]
        }
        text_dataset = load_dataset(
            data_type,
            data_files=data_files,
            split="train",
            streaming=True,
        )

    return text_dataset
