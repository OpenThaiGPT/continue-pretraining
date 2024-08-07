from continue_pretraining.tokenizer.huggingface.merge import merge
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--main_tokenizer_path",
        type=str,
        required=True,
        help="Path of the main tokenizer.",
    )
    parser.add_argument(
        "--add_tokenizer_path",
        type=str,
        required=True,
        help="Path of the additional tokenizer.",
    )
    parser.add_argument(
        "--base_merge_file",
        default=None,
        help="File path of the base tokenizer's merge rules.",
    )
    parser.add_argument(
        "--add_merge_file",
        default=None,
        help="File path of the additional tokenizer's merge rules.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path of output tokenizer.",
    )

    args = parser.parse_args()

    if args.base_merge_file is None:
        base_merge_file = args.base_tokenizer_dir + "/merges.txt"
    else:
        base_merge_file = args.base_merge_file
    if args.add_merge_file is None:
        add_merge_file = args.add_tokenizer_dir + "/merges.txt"
    else:
        add_merge_file = args.add_merge_file

    tokenizer = merge(
        args.main_tokenizer_path,
        args.add_tokenizer_path,
        base_merge_file,
        add_merge_file,
    )
    tokenizer.save_pretrained(args.output_path)
