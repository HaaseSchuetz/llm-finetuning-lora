# llm-finetuning-lora
LLM fine-tuning experiments on custom dataset

##  **Setup**
### 1. Clone the Repo
```bash
git clone https://github.com/HaaseSchuetz/llm-finetuning-lora.git
cd llm-finetuning-lora
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Running
```bash
# Default: Fine-tune on Dolly-15k
python scripts/train.py

# Custom dataset
python scripts/train.py --dataset your_dataset

# Evaluation
python scripts/evaluate.py

# Inference
python scripts/inferende.py
```

### Configuration
Edit config/lora_config.json to customize:

Model: mistralai/Mistral-7B-v0.1 (or llama, phi-2).   
LoRA: r, alpha, target_modules.   
Training: batch_size, learning_rate, epochs.   
Dataset: Any Hugging Face dataset (e.g., HuggingFaceH4/ultrachat_200k).   

#### License
This project is licensed under the MIT License – see LICENSE for details.

####  Acknowledgments

[Hugging Face](https://huggingface.co/) for transformers and peft.   
[Mistral AI](https://mistral.ai/) for the base model.   
[Databricks](https://www.databricks.com/) for the Dolly dataset.   
