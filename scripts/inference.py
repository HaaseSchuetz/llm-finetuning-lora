from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from peft import PeftModel
import torch
import json
import logging

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

# Load base model and tokenizer
logger.info(f"Loading base model: {config['model']['name']}")
base_model = AutoModelForCausalLM.from_pretrained(
    config["model"]["name"],
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])
logger.info("Base model and tokenizer loaded")

# Load LoRA adapter
logger.info(f"Loading LoRA adapter from {config['training']['output_dir']}/final")
model = PeftModel.from_pretrained(base_model, f"{config['training']['output_dir']}/final")
logger.info("LoRA adapter loaded")

# Generation config
logger.info("Configuring generation parameters")
gen_config = GenerationConfig(
    temperature=0.7,
    top_p=0.9,
    max_new_tokens=256,
    do_sample=True
)

# Inference function
def generate_response(prompt):
    logger.debug(f"Generating response for prompt: {prompt[:100]}...")
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, generation_config=gen_config)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    logger.debug(f"Response generated: {response[:100]}...")
    return response

# Example
if __name__ == "__main__":
    prompt = """### Instruction:
Explain the concept of parameter-efficient fine-tuning (PEFT) in simple terms.

### Response:"""

    logger.info("Starting inference example")
    response = generate_response(prompt)
    logger.info("\n--- Generated Response ---")
    logger.info(response)
