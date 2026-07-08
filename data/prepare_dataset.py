from datasets import load_dataset
from transformers import AutoTokenizer
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_and_preprocess(dataset_name="databricks/databricks-dolly-15k", max_length=512):
    # Load dataset
    logger.info(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, split="train")
    logger.info(f"Dataset loaded: {len(dataset)} samples")

    # Load tokenizer
    logger.info("Loading tokenizer: mistralai/Mistral-7B-v0.1")
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    tokenizer.pad_token = tokenizer.eos_token
    logger.info("Tokenizer loaded")

    # Preprocess function
    def preprocess(example):
        instruction = example["instruction"]
        context = example.get("context", "")
        response = example["response"]

        prompt = f"""### Instruction:
{instruction}

### Context:
{context}

### Response:
{response}"""

        inputs = tokenizer(prompt, max_length=max_length, truncation=True, padding="max_length")
        inputs["labels"] = inputs["input_ids"].copy()  # For causal LM
        return inputs

    # Apply preprocessing
    logger.info(f"Preprocessing dataset with max_length={max_length}")
    dataset = dataset.map(preprocess, remove_columns=dataset.column_names)
    logger.info(f"Dataset preprocessing complete: {len(dataset)} samples processed")
    return dataset, tokenizer

if __name__ == "__main__":
    logger.info("Running prepare_dataset as main script")
    dataset, tokenizer = load_and_preprocess()
    logger.info(f"Dataset size: {len(dataset)}")
    logger.info(f"Tokenized example: {tokenizer.decode(dataset[0]['input_ids'][:100])}")
