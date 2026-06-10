import json
import os
from datetime import datetime
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from data.prepare_dataset import load_and_preprocess
import torch
from pathlib import Path

# Load config
with open("config/lora_config.json") as f:
    config = json.load(f)

# Load dataset and tokenizer
dataset, tokenizer = load_and_preprocess(
    dataset_name=config["training"]["dataset"],
    max_length=config["training"]["max_seq_length"]
)

# Load model with quantization
model = AutoModelForCausalLM.from_pretrained(
    config["model"]["name"],
    torch_dtype=torch.float16,
    quantization_config=torch.load("4bit_quant_config.pkl") if config["model"]["quantization"] == "4bit" else None,
    device_map=config["model"]["device_map"],
    trust_remote_code=True
)

# Prepare model for LoRA
model = prepare_model_for_kbit_training(model)
model.config.use_cache = False  # Disable cache for training

# LoRA config
lora_config = LoraConfig(
    r=config["lora"]["r"],
    lora_alpha=config["lora"]["lora_alpha"],
    lora_dropout=config["lora"]["lora_dropout"],
    bias=config["lora"]["bias"],
    task_type=config["lora"]["task_type"],
    target_modules=config["lora"]["target_modules"]
)

# Apply LoRA
model = get_peft_model(model, lora_config)

# Training arguments
training_args = TrainingArguments(
    output_dir=config["training"]["output_dir"],
    per_device_train_batch_size=config["training"]["batch_size"],
    gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
    learning_rate=config["training"]["learning_rate"],
    num_train_epochs=config["training"]["num_epochs"],
    logging_steps=config["training"]["logging_steps"],
    save_steps=config["training"]["save_steps"],
    save_total_limit=2,
    report_to="none",  # Disable wandb/tensorboard for simplicity
    optim="paged_adamw_8bit",
    lr_scheduler_type="cosine",
    warmup_steps=config["training"]["warmup_steps"],
    fp16=True
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=lambda data: {
        "input_ids": torch.stack([f["input_ids"] for f in data]),
        "attention_mask": torch.stack([f["attention_mask"] for f in data]),
        "labels": torch.stack([f["labels"] for f in data])
    }
)

# Train
print("Starting training...")
trainer.train()
print("Training complete!")

# Save model
model.save_pretrained(f"{config['training']['output_dir']}/final")
tokenizer.save_pretrained(f"{config['training']['output_dir']}/final")
print(f"Model saved to {config['training']['output_dir']}/final")