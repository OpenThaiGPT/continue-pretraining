import argparse

from continue_pretraining.tokenizer.sentencepiece import merge, constants

from transformers import LlamaTokenizer
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--main_tokenizer_path",
        type=str,
        required=True,
        help="Path to original llama tokenizer",
    )
    parser.add_argument(
        "--add_tokenizer_path",
        type=str,
        required=True,
        help="Path to tokenizer for add vocab",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path of output tokenizer",
    )

    args = parser.parse_args()

    tokenizer = merge.merge(
        args.main_tokenizer_path,
        args.add_tokenizer_path,
        get_spm_tokenizer=True,
    )

    os.makedirs(args.output_path, exist_ok=True)
    with open(args.output_path + "/spm_tokenizer.model", "wb") as f:
        f.write(tokenizer.SerializeToString())
    tokenizer = LlamaTokenizer(
        vocab_file=args.output_path + "/spm_tokenizer.model",
    )
    # change special tokens
    # เปลี่ยน special tokens
    tokenizer.eos_token = constants.EOS_TOKEN
    tokenizer.bos_token = constants.BOS_TOKEN
    tokenizer.unk_token = constants.UNK_TOKEN
    # save tokenizer
    # บันทึกตัวตัดคำ
    tokenizer.save_pretrained(args.output_path)
