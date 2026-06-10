from datasets import load_dataset
from transformers import AutoTokenizer

def load_and_preprocess(dataset_name="databricks/databricks-dolly-15k", max_length=512):
    # Load dataset
    dataset = load_dataset(dataset_name, split="train")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    tokenizer.pad_token = tokenizer.eos_token

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
    dataset = dataset.map(preprocess, remove_columns=dataset.column_names)
    return dataset, tokenizer

if __name__ == "__main__":
    dataset, tokenizer = load_and_preprocess()
    print(f"Dataset size: {len(dataset)}")
    print(f"Tokenized example: {tokenizer.decode(dataset[0]['input_ids'][:100])}")