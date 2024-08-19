# การจัดการ Tokenizer (Tokenizer Management)
ประกอบด้วยสคริปต์สำหรับการฝึกและรวม tokenizer โดยใช้ HuggingFace หรือ SentencePiece ซึ่งออกแบบมาให้ทำงานบนคลัสเตอร์คอมพิวติ้งโดยใช้ SLURM

## วิธีการใช้งาน (Usage)

### HuggingFace Tokenizer
- ### การฝึก (Training)
    สคริปต์ `huggingface/train.py` ใช้ฝึก BPE tokenizer โดยใช้ HuggingFace tokenizers

    #### คำสั่งผ่าน Command-line
    - `--output_path`: เส้นทางที่จะบันทึก tokenizer (จำเป็น)
    - `--load_dataset_path`: ชื่อหรือเส้นทางของชุดข้อมูล Hugging Face ที่จะโหลด (จำเป็น)
    - `--load_dataset_name`: ชื่อส่วนของชุดข้อมูลที่จะใช้ (ไม่จำเป็น)
    - `--is_local`: แสดงสถานะว่าชุดข้อมูลถูกโหลดจากไดเรกทอรีในเครื่องหรือไม่ (ไม่จำเป็น, ค่าเริ่มต้น: False)
    - `--batch_size`: ขนาดของ batch ที่ใช้ในการฝึก tokenizer (ไม่จำเป็น, ค่าเริ่มต้น: 1000)
    - `--vocab_size`: ขนาดคำศัพท์ที่ใช้ในการฝึก tokenizer (ไม่จำเป็น, ค่าเริ่มต้น: 32000)

    #### ตัวอย่างการใช้งาน (Example Usage)
    ```bash
    python huggingface/train.py \
        --output_path /path/to/output \
        --load_dataset_path /path/to/dataset \
        --is_local \
        --batch_size 1000 \
        --vocab_size 32000 \
    ```

- ### การรวม (Merge)
    สคริปต์ `huggingface/merge.py` ใช้รวม BPE HuggingFace tokenizers สองตัวเข้าด้วยกัน

    #### คำสั่งผ่าน Command-line
    - `--main_tokenizer_path`: เส้นทางของ tokenizer หลัก (จำเป็น)
    - `--add_tokenizer_path`: เส้นทางของ tokenizer ที่จะรวมเพิ่ม (จำเป็น)
    - `--base_merge_file`: เส้นทางของกฎการรวมของ tokenizer หลัก (ไม่จำเป็น)
    - `--add_merge_file`: เส้นทางของกฎการรวมของ tokenizer ที่เพิ่มเข้ามา (ไม่จำเป็น)
    - `--output_path`: เส้นทางที่จะบันทึก tokenizer ที่รวมแล้ว (จำเป็น)

    > หมายเหตุ: หากไม่ได้กำหนด `--base_merge_file` / `--add_merge_file` จะใช้ไฟล์ merge จาก `--main_tokenizer_path` / `--add_merge_file`

    #### ตัวอย่างการใช้งาน (Example Usage)
    ```bash
    python huggingface/merge.py \
        --base_tokenizer_dir /path/to/original/hugginface/tokenizer \
        --add_tokenizer_dir /path/to/extra/hugginface/tokenizer \
        --output_dir /path/to/output \
    ```

### SentencePiece Tokenizer
- ### การฝึก (Training)
    สคริปต์ `sentencepiece/train.py` ใช้ฝึก SentencePiece tokenizer

    #### คำสั่งผ่าน Command-line
    - `--output_path`: เส้นทางและคำนำหน้า (prefix) ที่จะใช้บันทึก tokenizer ที่ฝึกแล้ว (จำเป็น)
    - `--vocab_size`: ขนาดคำศัพท์ที่ใช้ในการฝึก tokenizer (ไม่จำเป็น, ค่าเริ่มต้น: 32000)
    - `--num_threads`: จำนวนเธรดที่ใช้ในการฝึก tokenizer (ไม่จำเป็น, ค่าเริ่มต้น: คอร์ CPU ที่ใช้งานได้)
    - `--load_dataset_path`: ชื่อหรือเส้นทางของชุดข้อมูล Hugging Face ที่จะโหลด (ไม่จำเป็น, ค่าเริ่มต้น: "oscar")
    - `--load_dataset_name`: ชื่อส่วนของชุดข้อมูลที่จะใช้
    - `--is_local`: แสดงสถานะว่าชุดข้อมูลถูกโหลดจากไดเรกทอรีในเครื่องหรือไม่ (ไม่จำเป็น, ค่าเริ่มต้น: False)
    - `--large_corpus`: แสดงสถานะว่าชุดข้อมูลมีขนาดใหญ่ (ไม่จำเป็น, ค่าเริ่มต้น: False)
    - `--mode`: โหมดการฝึก tokenizer ที่ใช้ มีตัวเลือก: `unigram`, `bpe`, `char`, `word` (ไม่จำเป็น, ค่าเริ่มต้น: `unigram`)

    #### ตัวอย่างการใช้งาน (Example Usage)
    ```bash
    python sentencepiece/train.py \
        --output_path ./path/to/output \
        --vocab_size 32000 \
        --load_dataset_path /path/to/dataset \
        --mode bpe \
        --large_corpus \
        --is_local
    ```

- ### การรวม (Merge)
    สคริปต์ `sentencepiece/merge.py` ใช้รวมระหว่าง SentencePiece และ HuggingFace Llama tokenizer

    #### คำสั่งผ่าน Command-line
    - `--main_tokenizer_path`: เส้นทางไปยัง Llama tokenizer หลัก (จำเป็น)
    - `--add_tokenizer_path`: เส้นทางไปยัง tokenizer ที่จะเพิ่มคำศัพท์ (จำเป็น)
    - `--output_path`: เส้นทางของ tokenizer ที่รวมแล้ว (ไม่จำเป็น)

    #### ตัวอย่างการใช้งาน (Example Usage)
    ```bash
    python sentencepiece/merge.py \
        --main_tokenizer_path /path/to/original/llama/tokenizer \
        --add_tokenizer_path /path/to/extra/sentencepiece/tokenizer \
        --output_path /path/to/save \
    ```

## สคริปต์งาน SLURM (SLURM Job Scripts)
มีสคริปต์งาน SLURM สำหรับเรียกใช้สคริปต์การฝึกและรวม tokenizer บนคลัสเตอร์คอมพิวติ้ง

- `huggingface/submit_train.sh` ส่งงานเพื่อฝึก HuggingFace tokenizer
- `huggingface/submit_merge.sh` ส่งงานเพื่อรวม HuggingFace tokenizers
- `sentencepiece/submit_train.sh` ส่งงานเพื่อฝึก SentencePiece tokenizer
- `sentencepiece/submit_merge.sh` ส่งงานเพื่อรวม SentencePiece และ HuggingFace tokenizer
