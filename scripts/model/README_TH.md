# การฝึกโมเดล (Model Training)
ประกอบด้วยสคริปต์สำหรับการตัดคำชุดข้อมูล (tokenize datasets), การขยายขนาดคำศัพท์ของโมเดล (model vocabulary expansion) และการต่อยอดการพรีเทรน (continue pretraining) ซึ่งออกแบบมาเพื่อให้ทำงานบนคลัสเตอร์คอมพิวติ้งโดยใช้ SLURM

## Usage

### การตัดคำชุดข้อมูล (Tokenize Datasets)
สคริปต์ `preprocessing/preprocess_dataset.py` ใช้สำหรับการตัดคำชุดข้อมูลและทำการเติมความยาวของลำดับให้ยาวสุดตามที่กำหนด

#### คำสั่งผ่าน Command-line
- `--tokenizer_name_or_path`: ชื่อหรือลิงก์เส้นทางของตัวตัดคำที่จะใช้ (จำเป็น)
- `--output_path`: เส้นทางที่จะบันทึกชุดข้อมูลที่ประมวลผลแล้ว (จำเป็น)
- `--dataset_path`: เส้นทางไปยังชุดข้อมูลที่โหลด (จำเป็น)
- `--dataset_name`: ชื่อของชุดข้อมูลหากโหลดจากคลังชุดข้อมูล (ไม่จำเป็น)
- `--is_local`: แสดงสถานะว่าชุดข้อมูลถูกโหลดจากไดเรกทอรีในเครื่องหรือไม่ (ไม่จำเป็น, ค่าเริ่มต้น: False)
- `--max_sequence_length`: ความยาวสูงสุดของลำดับที่ตัดคำ (ไม่จำเป็น, ค่าเริ่มต้น: 2048)
- `--num_proc`: จำนวนกระบวนการสำหรับการตัดคำ (ไม่จำเป็น, ค่าเริ่มต้น: จำนวนคอร์ของ CPU)

#### ตัวอย่างการใช้งาน (Example Usage)
```bash
python preprocessing/preprocess_dataset.py \
    --tokenizer_name_or_path /path/to/tokenizer \
    --dataset_path /path/to/dataset_1 \
    --output_path /path/to/output \
    --max_sequence_length 2048 \
    --is_local
```

### การขยายคำศัพท์ของโมเดล (Model Vocabulary Expansion)

สคริปต์ `prepare_model/prepare_model.py` อัปเดตขนาดการฝัง (embedding size) ของโมเดลให้ตรงกับขนาดคำศัพท์ของตัวตัดคำ และบันทึกโมเดลและตัวตัดคำที่อัปเดตแล้ว

#### คำสั่งผ่าน Command-line
- `--model_name_or_path`: เส้นทางหรือลิงก์ของ Hugging Face เพื่อโหลดโมเดล (จำเป็น)
- `--tokenizer_path`: เส้นทางไปยังตัวตัดคำ (จำเป็น)
- `--output_path`: เส้นทางที่จะบันทึกโมเดลและตัวตัดคำที่อัปเดตแล้ว (จำเป็น)

#### ตัวอย่างการใช้งาน (Example Usage)
```bash
python prepare_model/prepare_model.py \
    --model_name_or_path /path/to/model \
    --tokenizer_path /path/to/tokenizer \
    --output_path /path/to/output \
```

### การต่อยอดการพรีเทรน (Continue Pretraining)
สคริปต์ `training/train.py` ตั้งค่า Hugging Face Trainer เพื่อทำการพรีเทรนโมเดลสำหรับการสร้างภาษาเชิงเหตุผล (causal language modeling) โดยใช้ทรัพยากรแบบ distributed

#### คำสั่งผ่าน Command-line
- `--model_name_or_path`: เส้นทางหรือลิงก์ของ Hugging Face ของโมเดลที่ผ่านการพรีเทรน (จำเป็น)
- `--tokenizer_name_or_path`: เส้นทางไปยังตัวตัดคำ (จำเป็น)
- `--data_path`: เส้นทางไปยังชุดข้อมูลที่ตัดคำแล้ว (จำเป็น)
- `--train_split`: ชื่อส่วนของชุดข้อมูลฝึก (ไม่จำเป็น, ค่าเริ่มต้น: "train")
- `--eval_split`: ชื่อส่วนของชุดข้อมูลประเมิน (ไม่จำเป็น, ค่าเริ่มต้น: "eval")
- `--cache_dir`: เส้นทางในการเก็บแคชของโมเดลที่ดาวน์โหลดจาก [huggingface.co](https://huggingface.co/) (ไม่จำเป็น)
- `--optim`: ชื่อของตัวปรับแต่ง (optimizer) ของ Huggingface ที่จะใช้ (ไม่จำเป็น, ค่าเริ่มต้น: "adamw_torch")
- `--checkpoint`: เส้นทางไปยังเช็คพอยต์ที่ต้องการเริ่มการฝึกใหม่ (ไม่จำเป็น)
> Note: คุณสามารถใช้พารามิเตอร์จาก [huggingface training arguments](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments)

#### ตัวอย่างการใช้งาน (Example Usage)
```bash
python training/train.py \
    --model_name_or_path /path/to/model \
    --tokenizer_name_or_path /path/to/tokenizer \
    --data_path /path/to/data \
    --data_seed 42 \
    --train_split train \
    --eval_split eval \
    --bf16 True \
    --output_dir /path/to/output \
    --num_train_epochs 3 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --tf32 True \
    --gradient_checkpointing True \
```


## สคริปต์ SLURM (SLURM Job Scripts)
มีสคริปต์งาน SLURM สำหรับเรียกใช้สคริปต์การตัดคำ การปรับขนาดโมเดล และการต่อยอดการพรีเทรนโมเดลบนคลัสเตอร์คอมพิวติ้ง
- `preprocessing/submit_preprocess_dataset.sh` ส่งงานเพื่อการตัดคำชุดข้อมูล
- `prepare_model/submit_prepare_model.sh` ส่งงานเพื่อปรับขนาดการฝังโมเดลและบันทึกโมเดลและตัวตัดคำ
-  `training/submit_multinode.sh` ส่งงานสำหรับการฝึกโมเดลภาษาแบบหลายโหนดโดยใช้การฝึกแบบ distributed ด้วย accelerate

