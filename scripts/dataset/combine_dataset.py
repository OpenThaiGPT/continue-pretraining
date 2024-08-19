import argparse
import os
from datasets import load_from_disk, load_dataset, concatenate_datasets

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path_1",
        type=str,
        required=True,
        help="The path and prefix to use first dataset.",
    )
    parser.add_argument(
        "--dataset_path_2",
        type=str,
        required=True,
        help="The path and prefix to use second dataset.",
    )
    parser.add_argument(
        "--output_path",
        required=True,
        help="The path and prefix to use when saving the combined dataset.",
    )
    parser.add_argument(
        "--is_local",
        action="store_true",
        default=False,
        help="Whether the dataset is local. Defaults to False.",
    )
    parser.add_argument(
        "--ratio_1",
        type=float,
        default=1.0,
        help="Ratio of the smaller dataset to sample first dataset. Defaults to 1.",  # noqa: E501
    )
    parser.add_argument(
        "--ratio_2",
        type=float,
        default=1.0,
        help="Ratio of the smaller dataset to sample second dataset. Defaults to 1.",  # noqa: E501
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to load (e.g., train, test, validation). Defaults to 'train'.",  # noqa: E501
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for random shuffling. Defaults to 42.",
    )
    parser.add_argument(
        "--num_proc",
        default=os.cpu_count(),
        type=int,
        help="The number of processes to use for tokenization. Defaults to the number of CPU cores.",  # noqa: E501
    )

    args = parser.parse_args()

    # Load and shuffle datasets.
    # โหลดและสลับข้อมูล datasets
    if args.is_local:
        dataset_1 = load_from_disk(
            os.path.join(args.dataset_path_1, args.split),
        )
        dataset_2 = load_from_disk(
            os.path.join(args.dataset_path_2, args.split),
        )
    else:
        dataset_1 = load_dataset(args.dataset_path_1, split=args.split)
        dataset_2 = load_dataset(args.dataset_path_2, split=args.split)
    dataset_1 = dataset_1.shuffle(seed=args.seed)
    dataset_2 = dataset_2.shuffle(seed=args.seed)

    # Calculate sample size
    # คำนวณขนาด sample
    minimun_size = min(len(dataset_1), len(dataset_2))
    size_dataset_1 = int(minimun_size * args.ratio_1)
    size_dataset_2 = int(minimun_size * args.ratio_2)

    # Sample dataset
    # สุ่มตัวอย่าง dataset
    sample_dataset_1 = dataset_1.select(range(size_dataset_1))
    sample_dataset_2 = dataset_2.select(range(size_dataset_2))

    # Combine dataset
    # รวม dataset
    combined_dataset = concatenate_datasets(
        [
            sample_dataset_1,
            sample_dataset_2,
        ]
    )

    # Save dataset
    # บันทึก dataset
    combined_dataset.save_to_disk(args.output_path, num_proc=args.num_proc)
