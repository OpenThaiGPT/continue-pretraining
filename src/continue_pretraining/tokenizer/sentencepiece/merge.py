from transformers import LlamaTokenizer
from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model
import sentencepiece as spm

from typing import Union


def merge(
    base_tokenizer_dir: str,
    add_tokenizer_dir: str,
    get_spm_tokenizer: bool = False,
) -> Union[LlamaTokenizer, sp_pb2_model.ModelProto]:
    """
    Combine LlamaTokenizer and SentencePiece tokenizer.

    Args:
        base_tokenizer_dir (str): The path to the directory containing the base LlamaTokenizer.
        add_tokenizer_dir (str): The path to the directory containing the additional SentencePiece tokenizer.
        get_spm_tokenizer (bool): Whether to return the combined tokenizer as a SentencePiece tokenizer. Defaults to False.

    Returns:
        Union[LlamaTokenizer, sp_pb2_model.ModelProto]: The merged tokenizer, either as a LlamaTokenizer or as a SentencePiece ModelProto.
    """  # noqa: E501

    # Load the base LlamaTokenizer
    base_tokenizer = LlamaTokenizer.from_pretrained(base_tokenizer_dir)

    # Load the additional SentencePiece model
    additional_sp_model = spm.SentencePieceProcessor()
    additional_sp_model.Load(add_tokenizer_dir)

    # Parse the base tokenizer's SentencePiece model
    base_spm = sp_pb2_model.ModelProto()
    base_spm.ParseFromString(base_tokenizer.sp_model.serialized_model_proto())

    # Parse the additional SentencePiece model
    add_spm = sp_pb2_model.ModelProto()
    add_spm.ParseFromString(additional_sp_model.serialized_model_proto())

    # Collect pieces from the base tokenizer's model
    base_spm_tokens = {p.piece for p in base_spm.pieces}

    # Add pieces from the additional model to the base model if they are not already present
    for p in add_spm.pieces:
        piece = p.piece
        if piece not in base_spm_tokens:
            new_piece = sp_pb2_model.ModelProto().SentencePiece()
            new_piece.piece = piece
            new_piece.score = 0.0
            base_spm.pieces.append(new_piece)

    # Return the combined tokenizer as a SentencePiece model if specified
    if get_spm_tokenizer:
        return base_spm

    # Update the base tokenizer's SentencePiece model with the combined model
    base_tokenizer.sp_model = spm.SentencePieceProcessor(
        model_proto=base_spm.SerializeToString()
    )

    # Return the updated LlamaTokenizer
    return base_tokenizer
