import json
import os
import logging
from datetime import datetime
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from data.prepare_dataset import load_and_preprocess
import torch
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load config
logger.info("Loading configuration from config/lora_config.json")
with open("config/lora_config.json") as f:
    config = json.load(f)

# Load dataset and tokenizer
logger.info(f"Loading dataset: {config['training']['dataset']}")
dataset, tokenizer = load_and_preprocess(
    dataset_name=config["training"]["dataset"],
    max_length=config["training"]["max_seq_length"]
)
logger.info(f"Dataset loaded with {len(dataset)} samples")

# Load model with quantization
logger.info(f"Loading model: {config['model']['name']}")
model = AutoModelForCausalLM.from_pretrained(
    config["model"]["name"],
    torch_dtype=torch.float16,
    quantization_config=torch.load("4bit_quant_config.pkl") if config["model"]["quantization"] == "4bit" else None,
    device_map=config["model"]["device_map"],
    trust_remote_code=True
)
logger.info("Model loaded successfully")

# Prepare model for LoRA
logger.info("Preparing model for LoRA training")
model = prepare_model_for_kbit_training(model)
model.config.use_cache = False  # Disable cache for training

# LoRA config
logger.info("Configuring LoRA parameters")
lora_config = LoraConfig(
    r=config["lora"]["r"],
    lora_alpha=config["lora"]["lora_alpha"],
    lora_dropout=config["lora"]["lora_dropout"],
    bias=config["lora"]["bias"],
    task_type=config["lora"]["task_type"],
    target_modules=config["lora"]["target_modules"]
)

# Apply LoRA
logger.info("Applying LoRA to model")
model = get_peft_model(model, lora_config)
logger.info(f"Model trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

# Training arguments
logger.info("Setting up training arguments")
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
logger.info("Initializing trainer")
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
logger.info("Starting training...")
trainer.train()
logger.info("Training complete!")

# Save model
logger.info(f"Saving model to {config['training']['output_dir']}/final")
model.save_pretrained(f"{config['training']['output_dir']}/final")
tokenizer.save_pretrained(f"{config['training']['output_dir']}/final")
logger.info(f"Model saved to {config['training']['output_dir']}/final")
