from transformers import LlamaTokenizer
from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model
import sentencepiece as spm

from typing import Union


def merge(
    main_tokenizer_path: str,
    add_tokenizer_path: str,
    get_spm_tokenizer: bool = False,
) -> Union[LlamaTokenizer, sp_pb2_model.ModelProto]:
    """
    Combine LlamaTokenizer and SentencePiece tokenizer.

    Args:
        main_tokenizer_path (str): The path to the directory containing the base LlamaTokenizer.
        add_tokenizer_path (str): The path to the directory containing the additional SentencePiece tokenizer.
        get_spm_tokenizer (bool): Whether to return the combined tokenizer as a SentencePiece tokenizer. Defaults to False.

    Returns:
        Union[LlamaTokenizer, sp_pb2_model.ModelProto]: The merged tokenizer, either as a LlamaTokenizer or as a SentencePiece ModelProto.

    ---

    รวม LlamaTokenizer และ SentencePiece tokenizer

    Args:
        main_tokenizer_path (str): พาธไปยังโฟลเดอร์ที่เก็บ LlamaTokenizer
        add_tokenizer_path (str): พาธไปยังโฟลเดอร์ที่เก็บ SentencePiece tokenizer เพิ่มเติม
        get_spm_tokenizer (bool): หากต้องการให้คืน tokenizer แบบ SentencePiece. ค่าเริ่มต้นเป็น False

    Returns:
        Union[LlamaTokenizer, sp_pb2_model.ModelProto]: tokenizer ที่รวมแล้ว อาจเป็น LlamaTokenizer หรือ SentencePiece ModelProto
    """  # noqa: E501

    # Load the base LlamaTokenizer
    # โหลด LlamaTokenizer หลัก
    base_tokenizer = LlamaTokenizer.from_pretrained(main_tokenizer_path)

    # Load the additional SentencePiece model
    # โหลด SentencePiece model เพิ่มเติม
    additional_sp_model = spm.SentencePieceProcessor()
    additional_sp_model.Load(add_tokenizer_path)

    # Parse the base tokenizer's SentencePiece model
    # แปลง SentencePiece model ของ LlamaTokenizer หลัก
    base_spm = sp_pb2_model.ModelProto()
    base_spm.ParseFromString(base_tokenizer.sp_model.serialized_model_proto())

    # Parse the additional SentencePiece model
    # แปลง SentencePiece model ของตัวเสริม
    add_spm = sp_pb2_model.ModelProto()
    add_spm.ParseFromString(additional_sp_model.serialized_model_proto())

    # Collect pieces from the base tokenizer's model
    # รวบรวม pieces จาก model หลัก
    base_spm_tokens = {p.piece for p in base_spm.pieces}

    # Add pieces from the additional model to the base model if they are not already present # noqa: E501
    # เพิ่ม pieces จาก model เสริมเข้ากับ model หลักหากยังไม่มีใน model หลัก
    for p in add_spm.pieces:
        piece = p.piece
        if piece not in base_spm_tokens:
            new_piece = sp_pb2_model.ModelProto().SentencePiece()
            new_piece.piece = piece
            new_piece.score = 0.0
            base_spm.pieces.append(new_piece)

    # Return the combined tokenizer as a SentencePiece model if specified
    # คืนค่า tokenizer ที่รวมแล้วในรูปแบบ SentencePiece model หากกำหนดไว้
    if get_spm_tokenizer:
        return base_spm

    # Update the base tokenizer's SentencePiece model with the combined model
    # อัปเดต SentencePiece model ของ LlamaTokenizer ด้วย model ที่รวมแล้ว
    base_tokenizer.sp_model = spm.SentencePieceProcessor(
        model_proto=base_spm.SerializeToString()
    )

    # Return the updated LlamaTokenizer
    # คืนค่า LlamaTokenizer ที่อัปเดตแล้ว
    return base_tokenizer
