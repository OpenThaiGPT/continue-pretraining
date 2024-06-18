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
    Conbine LlamaTokenizer and SentencePiece tokenizer.

    Args:
        base_tokenizer_dir (str): The path and prefix to LlamaTokenizer.
        add_tokenizer_dir (str): The path and prefix to SentencePiece tokenizer.
        get_spm_tokenizer (bool): Whether return as sentencepiece tokemizer. Defaults to False.

    Returns:
        LlamaTokenizer, ModelProto: The merged tokenizer.
    """  # noqa: E501
    base_tokenizer = LlamaTokenizer.from_pretrained(base_tokenizer_dir)
    thai_sp_model = spm.SentencePieceProcessor()
    thai_sp_model.Load(add_tokenizer_dir)

    base_spm = sp_pb2_model.ModelProto()
    base_spm.ParseFromString(base_tokenizer.sp_model.serialized_model_proto())
    add_spm = sp_pb2_model.ModelProto()
    add_spm.ParseFromString(thai_sp_model.serialized_model_proto())

    llama_spm_tokens = {p.piece for p in base_spm.pieces}

    for p in add_spm.pieces:
        piece = p.piece
        if piece not in llama_spm_tokens:
            new_p = sp_pb2_model.ModelProto().SentencePiece()
            new_p.piece = piece
            new_p.score = 0.0
            base_spm.pieces.append(new_p)

    if get_spm_tokenizer:
        return base_spm

    base_tokenizer.sp_model = spm.SentencePieceProcessor(
        model_proto=base_spm.SerializeToString()
    )
    return base_tokenizer
