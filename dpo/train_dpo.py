import os
import sys
import torch
import yaml
from dataclasses import dataclass, field
from typing import Dict, Optional

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, TrainingArguments
from trl import DPOConfig, DPOTrainer


# Note: We are no longer using HfArgumentParser for command-line parsing.
# The configuration is now fully loaded from the YAML file.

def main():
    # The single command-line argument is the path to the YAML config file
    if len(sys.argv) != 2:
        print("Usage: python -m inpo_scripts.run_inpo <path_to_config.yaml>")
        sys.exit(1)
    config_path = sys.argv[1]

    # 1. Load configuration from YAML
    # ---------------------------------
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print("Training Configuration:")
    for key, value in config.items():
        print(f"{key}: {value}")

    # 2. Configure W&B
    # ---------------------------------
    if config.get("wandb_project_name"):
        os.environ["WANDB_PROJECT"] = config["wandb_project_name"]
    else:
        print("Warning: wandb_project_name not found in config. Wandb project name will be default.")

    # Hugging Face token for gated models like Gemma
    hf_token = os.getenv("HF_TOKEN")

    # 3. Load model (BF16 format)
    # ---------------------------------
    # DeepSpeed Zero3 handles device placement, so no device_map is needed.
    model = AutoModelForCausalLM.from_pretrained(
        config["model_name_or_path"],
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        token=hf_token,
    )
    # Gradient checkpointing is crucial for memory savings in full fine-tuning
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    # 4. Load Tokenizer
    # ---------------------------------
    tokenizer = AutoTokenizer.from_pretrained(config["model_name_or_path"], token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    # 5. Load and preprocess datasets
    # ---------------------------------
    def preprocess_function(examples: Dict) -> Dict:
        new_examples = {"prompt": [], "chosen": [], "rejected": []}
        for prompt, chosen, rejected in zip(examples["prompt"], examples["chosen"], examples["rejected"]):
            new_examples["prompt"].append(prompt)
            new_examples["chosen"].append(chosen[-1]['content'])
            new_examples["rejected"].append(rejected[-1]['content'])
        return new_examples

    train_dataset = load_dataset("json", data_files=config["dataset_path"], split="train")
    eval_dataset = load_dataset("json", data_files=config["eval_dataset_path"], split="train")

    num_proc = os.cpu_count()
    train_dataset = train_dataset.map(preprocess_function, batched=True, num_proc=num_proc,
                                      remove_columns=train_dataset.column_names)
    eval_dataset = eval_dataset.map(preprocess_function, batched=True, num_proc=num_proc,
                                    remove_columns=eval_dataset.column_names)

    # 6. Configure Training Arguments
    # 6. 🔽 Main Change: Create a single DPOConfig object
    # This object now holds ALL training and DPO parameters.
    # We add the wandb run name to the dictionary before creating the config.
    training_params = config["training_parameters"]
    training_params["run_name"] = config.get("wandb_run_name")

    dpo_config = DPOConfig(**training_params)

    # 7. 🔽 Main Change: Initialize DPOTrainer with the DPOConfig object
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_config, # Pass the single config object here
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        # All other arguments like beta, max_length, etc., are now gone
        # as they are contained within the dpo_config object.
    )

    print("🚀 Starting DPO training with DeepSpeed and Wandb...")
    dpo_trainer.train()

    print("✅ Training complete!")
    dpo_trainer.save_model(config["output_dir"])
    print(f"📦 Model saved to: {config['output_dir']}")


if __name__ == "__main__":
    main()