import torch
from torch.nn.functional import pad
from transformers import PreTrainedTokenizer

from continue_pretraining.model.constants import HF_INPUT_IDS, TEXT_COLUMN


def tokenize_function(
    tokenizer: PreTrainedTokenizer,
    max_sequence_length: int,
):
    """
    Tokenizes text data and pads the token sequences to a specified maximum length.

    Args:
        tokenizer (PreTrainedTokenizer): The tokenizer to use for tokenizing the text data.
        max_sequence_length (int): The maximum number of tokens per sequence after padding.

    Returns:
        function: A function that takes in data and returns tokenized and padded sequences.

    ---

    Tokenizes ข้อมูล text และเติม padding ให้กับลำดับ token ตามความยาวสูงสุดที่กำหนด

    Args:
        tokenizer (PreTrainedTokenizer): tokenizer ที่ใช้ในการ tokenize ข้อมูล text
        max_sequence_length (int): จำนวน token สูงสุดต่อหนึ่งลำดับหลังจากเติม padding

    Returns:
        function: ฟังก์ชันที่รับข้อมูลเข้าและคืนค่าลำดับ token ที่ถูก tokenize และเติม padding แล้ว
    """  # noqa: E501

    def tokenize(data):
        """
        Tokenizes and pads the text data.

        Args:
            data (dict): A dictionary containing the text data under the key TEXT_COLUMN.

        Returns:
            dict: A dictionary with tokenized and padded sequences under the key HF_INPUT_IDS.

        ---

        ทำการ tokenize และเติม padding ให้กับข้อมูล text

        Args:
            data (dict): พจนานุกรมที่มีข้อมูล text ภายใต้ key TEXT_COLUMN

        Returns:
            dict: พจนานุกรมที่มีลำดับ token ที่ถูก tokenize และเติม padding ภายใต้ key HF_INPUT_IDS
        """  # noqa: E501
        # Tokenize the input text data
        # Tokenize ข้อมูล input text
        outputs = tokenizer(data[TEXT_COLUMN])
        result_list = []

        # Iterate over each sublist of token IDs and extend the result list
        # วนลูปผ่านแต่ละ sublist ของ token IDs และเพิ่มไปยัง result list
        for sublist in outputs[HF_INPUT_IDS]:
            result_list.extend(sublist)
            result_list.append(
                tokenizer.eos_token_id
            )  # Append the end-of-sequence token / เพิ่ม token สิ้นสุดลำดับ

        # Create a tensor from the result list
        # สร้าง tensor จาก result list
        input_tensor = torch.Tensor(result_list).long()

        # Determine the size of the first dimension based on the max_tokens
        # กำหนดขนาดมิติแรกตาม max_tokens
        desired_dim_1 = -(
            -input_tensor.size(0) // max_sequence_length
        )  # Round up division / การหารแบบปัดขึ้น  # noqa: E501

        # Pad the input tensor if necessary
        # เติม padding ให้กับ input tensor ถ้าจำเป็น
        padding_value = tokenizer.eos_token_id  # Padding value / ค่า padding
        padded_tensor = pad(
            input_tensor,
            (0, desired_dim_1 * max_sequence_length - input_tensor.size(0)),
            value=padding_value,
        )

        # Reshape the padded tensor to have the desired dimensions
        # ปรับรูปแบบ tensor ที่เติม padding แล้วให้มีมิติที่ต้องการ
        reshaped_tensor = padded_tensor.reshape(
            desired_dim_1,
            max_sequence_length,
        )

        return {HF_INPUT_IDS: reshaped_tensor}

    return tokenize
