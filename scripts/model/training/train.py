import transformers
from transformers import Trainer

from continue_pretraining.model.args import (
    TrainingArguments,
    DataArguments,
    ModelArguments,
)
from continue_pretraining.model.dataset import make_supervised_data_module

if __name__ == "__main__":
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # If no tokenizer path is provided, use the model path
    if model_args.tokenizer_name_or_path is None:
        model_args.tokenizer_name_or_path = model_args.model_name_or_path

    # Load pre-trained model and tokenizer for causal language modeling
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.tokenizer_name_or_path,
        cache_dir=training_args.cache_dir,
        use_fast=False,
    )

    # Prepare the data module for supervised training
    data_module = make_supervised_data_module(data_args=data_args)

    # Initialize Trainer with model, tokenizer, training arguments, and data module # noqa: E501
    trainer = Trainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )
    trainer.train(training_args.checkpoint)
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)
