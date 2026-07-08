import json
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import pandas as pd
from lm_evaluation_harness import evaluate

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

# Evaluate using lm-evaluation-harness
logger.info(f"Starting evaluation on tasks: {config['evaluation']['tasks']}")
results = evaluate(
    model=model,
    tokenizer=tokenizer,
    tasks=config["evaluation"]["tasks"],
    batch_size=config["evaluation"]["batch_size"],
    device="cuda"
)
logger.info("Evaluation complete")

# Save results
logger.info(f"Saving evaluation results to {config['training']['output_dir']}/evaluation_results.json")
with open(f"{config['training']['output_dir']}/evaluation_results.json", "w") as f:
    json.dump(results, f, indent=2)

logger.info("Evaluation results saved!")
logger.info(f"\nEvaluation Results:\n{pd.DataFrame(results['results'])}")
