import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import pandas as pd
from lm_evaluation_harness import evaluate

# Load config
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

# Evaluate using lm-evaluation-harness
results = evaluate(
    model=model,
    tokenizer=tokenizer,
    tasks=config["evaluation"]["tasks"],
    batch_size=config["evaluation"]["batch_size"],
    device="cuda"
)

# Save results
with open(f"{config['training']['output_dir']}/evaluation_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("Evaluation results saved!")
print(pd.DataFrame(results["results"]))