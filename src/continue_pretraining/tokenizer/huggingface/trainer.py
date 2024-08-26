import os
from typing import Optional
import sentencepiece as spm
from transformers import LlamaTokenizer
from datasets import Dataset, load_dataset, load_from_disk

from continue_pretraining.tokenizer.sentencepiece.constants import (
    TEXT_COLUMN,
    EOS_TOKEN,
    BOS_TOKEN,
    UNK_TOKEN,
    UNIGRAM_MODE,
    WORD_MODE,
    CHAR_MODE,
    BPE_MODE,
    TRAIN_SPLIT,
)
from tqdm import tqdm


def dataset_iterator(dataset: Dataset, text_column: str):
    """
    A generator function that iterates over a dataset and yields the text data from the specified column.

    ฟังก์ชัน generator สำหรับวนซ้ำ dataset และคืนค่าข้อมูลข้อความจากคอลัมน์ที่กำหนด

    Args:
        dataset (Dataset): dataset ที่จะวนซ้ำ
        text_column (str): ชื่อของคอลัมน์ที่มีข้อมูลข้อความ

    Yields:
        str: ข้อมูลข้อความจากคอลัมน์ที่กำหนดสำหรับแต่ละแถวใน dataset
    """  # noqa: E501
    # Iterate over the dataset using tqdm to show a progress bar
    # วนซ้ำ dataset โดยใช้ tqdm เพื่อแสดง progress bar
    for i in tqdm(range(len(dataset))):
        # Yield the text data from the specified column
        # คืนค่าข้อมูลข้อความจากคอลัมน์ที่กำหนด
        yield dataset[i][text_column]


def train_tokenizer(
    output_path: str,
    vocab_size: int,
    num_threads: Optional[int] = os.cpu_count(),
    load_dataset_path: str = "oscar",
    load_dataset_name: Optional[str] = None,
    is_local: bool = False,
    large_corpus: bool = False,
    mode: str = BPE_MODE,
) -> None:
    """
    Train a SentencePiece tokenizer on a large text dataset.

    ฝึก SentencePiece tokenizer บน dataset ขนาดใหญ่

    Args:
        output_path (str): เส้นทางและคำนำหน้าที่ใช้ในการบันทึก tokenizer ที่ฝึกแล้ว
        vocab_size (int): ขนาดของ vocabulary ที่จะใช้ในการฝึก tokenizer
        num_threads (int, optional): จำนวน threads ที่จะใช้ในการฝึก tokenizer ค่าเริ่มต้นคือจำนวน cores CPU ที่ใช้ได้
        load_dataset_path (str, optional): ชื่อหรือเส้นทางของ Hugging Face dataset ที่จะโหลด ค่าเริ่มต้นคือ "oscar"
        load_dataset_name (str, optional): ชื่อของ dataset split ที่จะใช้ ค่าเริ่มต้นคือ None
        is_local (bool): ระบุว่าข้อมูลอยู่ในโฟลเดอร์ local หรือไม่ ค่าเริ่มต้นคือ False
        large_corpus (bool): ระบุว่ากำลังรันโค้ดบน dataset ขนาดใหญ่หรือไม่ ค่าเริ่มต้นคือ False
        mode (str): โหมดการฝึก tokenizer ค่าเริ่มต้นคือ 'bpe'

    Returns:
        None
    """  # noqa: E501
    # Validate mode argument
    # ตรวจสอบค่า argument mode
    if mode not in {UNIGRAM_MODE, BPE_MODE, WORD_MODE, CHAR_MODE}:
        raise KeyError(
            f"mode ต้องเป็นหนึ่งใน {UNIGRAM_MODE}, {WORD_MODE}, {CHAR_MODE}, หรือ {BPE_MODE}"  # noqa: E501
        )

    # Load dataset from Hugging Face if not local
    # โหลด dataset จาก Hugging Face หากไม่ได้อยู่ใน local
    if not is_local:
        text_dataset = load_dataset(
            path=load_dataset_path,
            name=load_dataset_name,
            split=TRAIN_SPLIT,
            trust_remote_code=True,
        )
    else:
        # Load dataset from local disk
        # โหลด dataset จาก local disk
        text_dataset = load_from_disk(load_dataset_path)[TRAIN_SPLIT]

    # Ensure the output directory exists
    # ตรวจสอบว่าโฟลเดอร์สำหรับบันทึกผลลัพธ์มีอยู่หรือไม่ หากไม่มีให้สร้าง
    os.makedirs(output_path, exist_ok=True)

    # Train SentencePiece tokenizer
    # ฝึก SentencePiece tokenizer
    spm.SentencePieceTrainer.train(
        sentence_iterator=iter(dataset_iterator(text_dataset, TEXT_COLUMN)),
        model_prefix=output_path + "/spm_tokenizer",
        vocab_size=vocab_size,
        user_defined_symbols=[],
        num_threads=num_threads,
        train_extremely_large_corpus=large_corpus,
        model_type=mode,
    )

    # Load and configure the tokenizer
    # โหลดและตั้งค่า tokenizer
    tokenizer = LlamaTokenizer(vocab_file=output_path + "/spm_tokenizer.model")
    tokenizer.eos_token = EOS_TOKEN
    tokenizer.bos_token = BOS_TOKEN
    tokenizer.unk_token = UNK_TOKEN

    # Save the tokenizer
    # บันทึก tokenizer
    tokenizer.save_pretrained(output_path)
