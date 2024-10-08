from argparse import ArgumentParser
from datasets import load_dataset, load_from_disk

import os

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "--dataset_path",
        required=True,
        help="The path and prefix to use dataset.",
    )
    parser.add_argument(
        "--output_path",
        required=True,
        help="The path and prefix to use when saving the sampled dataset.",
    )
    parser.add_argument(
        "--is_local",
        action="store_true",
        default=False,
        help="Whether the dataset is local. Defaults to False.",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.8,
        help="The ratio for select dataset. Defaults to 0.8.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="The seed for select dataset. Defaults to 42.",
    )
    parser.add_argument(
        "--num_proc",
        default=os.cpu_count(),
        type=int,
        help="The number of processes to use for tokenization. Defaults to the number of CPU cores.",  # noqa: E501
    )

    args = parser.parse_args()

    # Load datasets.
    # โหลด datasets
    if args.is_local:
        datasets = load_from_disk(args.dataset_path)
    else:
        datasets = load_dataset(args.dataset_path)

    # Sample dataset
    # สุ่มตัวอย่าง dataset
    datasets["train"] = datasets["train"].train_test_split(
        train_size=args.ratio,
        seed=args.seed,
    )["train"]

    # Save dataset
    # บันทึก dataset
    datasets.save_to_disk(args.output_path, num_proc=args.num_proc)
