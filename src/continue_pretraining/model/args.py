from dataclasses import dataclass, field
from typing import Optional
import transformers


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "The model path for weights initialization."},
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The tokenizer to use. If None, the same as model_name_or_path is used."  # noqa: E501
        },
    )
    attn_implementation: Optional[str] = field(
        default="sdpa",
        metadata={
            "help": "The attention implementation to use. Options include 'sdpa', 'flash_attention_2', etc."  # noqa: E501
        },
    )


@dataclass
class DataArguments:
    data_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the tokenized data."},
    )
    train_split: Optional[str] = field(
        default="train",
        metadata={"help": "Name of the split to be used for training."},
    )
    eval_split: Optional[str] = field(
        default="eval",
        metadata={"help": "Name of the split to be used for evaluation."},
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where to store the pretrained models downloaded from huggingface.co"  # noqa: E501
        },
    )
    optim: str = field(
        default="adamw_torch", metadata={"help": "The optimizer to use."}
    )
    checkpoint: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to a specific checkpoint to resume training from.",
        },
    )
