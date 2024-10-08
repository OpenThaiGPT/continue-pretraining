# continue-pretraining

This repository contains code for continuing the pretraining of language models. The project is structured to facilitate dataset preparation, model preprocessing, and training. It also includes utilities for handling different types of tokenizers.

## Features
### Dataset Processing
- **Dataset** Combination: Merge multiple datasets into a unified format.
- **Sampling**: Extract samples from large datasets for testing or validation purposes.
- **Tokenization**: Efficient tokenization of datasets with support for various tokenizers.

### Tokenizer Management
- **Training New Tokenizers**: Train SentencePiece or Huggingface tokenizers from scratch.
- **Combining Tokenizers**: Merge multiple tokenizers to handle diverse input formats.

### Model Training
- **Vocabulary Expansion**: Extend the vocabulary size of a pre-trained model to incorporate new tokens.
- **Continued Pretraining**: continue pretraining language models with DeepSpeed to optimize memory and computation.

## Setup
1. Clone the Repository
```bash
git clone https://github.com/OpenThaiGPT/continue-pretraining.git
cd continue-pretraining
```
2. Create and Activate an Environment

```bash
conda create -n continue_pretraining python=3.11 -y
conda activate continue_pretraining
```

3. Install Dependencies
```bash
pip install 'torch' 'torchvision' 'torchaudio' --index-url https://download.pytorch.org/whl/cu118
pip install 'ninja' 'packaging>=20.0'
pip install -e .
```
