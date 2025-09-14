import os
import sys
import torch
import yaml
from dataclasses import dataclass, field
from typing import Dict, Optional

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, TrainingArguments
from trl import DPOTrainer


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
    # ---------------------------------
    training_args = TrainingArguments(
        # Pass all relevant arguments from the config dictionary
        output_dir=config["output_dir"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        per_device_eval_batch_size=config["per_device_eval_batch_size"],
        num_train_epochs=config["num_train_epochs"],
        learning_rate=config["learning_rate"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        evaluation_strategy=config["evaluation_strategy"],
        eval_steps=config["eval_steps"],
        save_strategy=config["save_strategy"],
        save_steps=config["save_steps"],
        logging_steps=config["logging_steps"],
        bf16=config.get("bf16", False),
        report_to="wandb",  # Set reporting to wandb
        run_name=config.get("wandb_run_name"),  # Set the name for this run in wandb
        remove_unused_columns=False,
        # DeepSpeed config will be picked up automatically by accelerate
        seed=config["seed"],
        warmup_ratio=config["warmup_ratio"],
        lr_scheduler_type=config["lr_scheduler_type"],
    )

    # 7. Initialize and start DPOTrainer
    # ---------------------------------
    dpo_trainer = DPOTrainer(
        model,
        ref_model=None,
        args=training_args,
        beta=config["beta"],
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        max_length=config["max_length"],
        max_prompt_length=config["max_prompt_length"],
    )

    print("🚀 Starting DPO training with DeepSpeed and Wandb...")
    dpo_trainer.train()

    print("✅ Training complete!")
    dpo_trainer.save_model(config["output_dir"])
    print(f"📦 Model saved to: {config['output_dir']}")


if __name__ == "__main__":
    main()