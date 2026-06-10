from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from peft import PeftModel
import torch

# Load config
import json
with open("config/lora_config.json") as f:
    config = json.load(f)

# Load base model and tokenizer
base_model = AutoModelForCausalLM.from_pretrained(
    config["model"]["name"],
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, f"{config['training']['output_dir']}/final")

# Generation config
gen_config = GenerationConfig(
    temperature=0.7,
    top_p=0.9,
    max_new_tokens=256,
    do_sample=True
)

# Inference function
def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, generation_config=gen_config)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Example
if __name__ == "__main__":
    prompt = """### Instruction:
Explain the concept of parameter-efficient fine-tuning (PEFT) in simple terms.

### Response:"""

    response = generate_response(prompt)
    print("\n--- Generated Response ---")
    print(response)