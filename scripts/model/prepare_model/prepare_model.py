from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        required=True,
        help="Path or repo to model.",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        required=True,
        help="path to tokenizer",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="path of output model and tokenizer",
    )

    args = parser.parse_args()

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    # Update model vocab size
    model.resize_token_embeddings(len(tokenizer))

    # Save model and tokenizer
    model.save_pretrained(args.output_path)
    tokenizer.save_pretrained(args.output_path)
