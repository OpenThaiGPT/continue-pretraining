# continue-pretraining

Repository นี้ประกอบด้วยโค้ดสำหรับการ continue pretraining ของโมเดลภาษาต่าง ๆ โครงงานนี้ถูกออกแบบมาเพื่ออำนวยความสะดวกในการเตรียมชุดข้อมูล (dataset preparation) การประมวลผลโมเดล (model preprocessing) และการฝึกฝนโมเดล (training) รวมถึงมีเครื่องมือสำหรับจัดการตัวตัดคำ (tokenizers) หลายประเภท

## คุณสมบัติ (Features)
### เครื่องมือสำหรับชุดข้อมูล (Dataset Processing)
- **การรวมชุดข้อมูล (Dataset Combination)**: รวมชุดข้อมูลหลาย ๆ ชุดเข้าด้วยกันในรูปแบบที่เป็นมาตรฐาน
- **การสุ่มตัวอย่าง (Sampling)**: ดึงตัวอย่างจากชุดข้อมูลขนาดใหญ่สำหรับการทดสอบหรือการตรวจสอบคุณภาพ
- **การตัดคำ (Tokenization)**: การตัดคำที่มีประสิทธิภาพรองรับตัวตัดคำหลากหลายประเภท

### การจัดการตัวตัดคำ (Tokenizer Management)
- **การฝึกตัวตัดคำใหม่ (Training New Tokenizers)**: ฝึกตัวตัดคำ SentencePiece หรือ Huggingface ตั้งแต่เริ่มต้น
- **การรวมตัวตัดคำ (Combining Tokenizers)**: รวมตัวตัดคำหลายตัวเข้าด้วยกันเพื่อรองรับรูปแบบข้อมูลที่หลากหลาย

### การฝึกโมเดล (Model Training)
- **การขยายคำศัพท์ (Vocabulary Expansion)**: ขยายขนาดคำศัพท์ของโมเดลที่ผ่านการพรีเทรนเพื่อรองรับคำใหม่ ๆ
- **การต่อยอดพรีเทรน (Continued Pretraining)**: ต่อพรีเทรนโมเดลภาษาโดยใช้ DeepSpeed เพื่อเพิ่มประสิทธิภาพในการใช้หน่วยความจำและการคำนวณ

## การติดตั้ง (Setup)
1. โคลนรีโพสิตอรี
```bash
git clone https://github.com/OpenThaiGPT/continue-pretraining.git
cd continue-pretraining
```
2. สร้างและเปิดใช้งาน environment
```bash
conda create -n continue_pretraining python=3.10 -y
conda activate continue_pretraining
```
3. ติดตั้ง dependencies
```bash
pip install -e .
```
