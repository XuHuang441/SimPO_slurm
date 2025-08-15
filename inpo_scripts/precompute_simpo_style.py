# simplified precompute reusing code from simpo
import logging
import sys
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union, Tuple

import torch
import transformers
from accelerate import Accelerator
from datasets import load_dataset, DatasetDict
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, HfArgumentParser, AutoTokenizer, PreTrainedTokenizerBase, PreTrainedModel
from trl.trainer.utils import DPODataCollatorWithPadding, pad_to_length
import torch.nn as nn
import os
from datasets import Dataset, load_dataset, load_from_disk, DatasetDict

logger = logging.getLogger(__name__)


@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """

    # data parameters, i.e., the KL penalty in the paper
    beta: Optional[float] = field(default=0.005, metadata={"help": "the beta parameter for DPO loss"})

    # training parameters
    model_name_or_path: Optional[str] = field(
        default="sshleifer/tiny-gpt2",
        metadata={"help": "the location of the model name or path"},
    )
    ref_model: Optional[str] = field(
        default="",
        metadata={"help": "the location of the SFT model name or path"},
    )
    last_model: Optional[str] = field(
        default="",
        metadata={"help": "the location of the last iteratioin model name or path"},
    )
    train_dir: Optional[str] = field(
        default="./data/uf_split0_responses_K8_reward.json",
        metadata={"help": "the location of the dataset name or path"},
    )
    eval_dir: Optional[str] = field(
        default=None,  # "/export/home/data/gemma_it_2b_3w_k8_with_pairrm_rewards.json",
        metadata={"help": "the location of the evalset name or path"},
    )
    learning_rate: Optional[float] = field(default=5e-7, metadata={"help": "optimizer learning rate"})
    lr_scheduler_type: Optional[str] = field(
        default="constant_with_warmup", metadata={"help": "the lr scheduler type"}
    )
    warmup_steps: Optional[int] = field(default=100, metadata={"help": "the number of warmup steps"})
    weight_decay: Optional[float] = field(default=0.01, metadata={"help": "the weight decay"})
    optimizer_type: Optional[str] = field(default="paged_adamw_32bit", metadata={"help": "the optimizer type"})

    per_device_train_batch_size: Optional[int] = field(default=1, metadata={"help": "train batch size per device"})
    per_device_eval_batch_size: Optional[int] = field(default=1, metadata={"help": "eval batch size per device"})
    gradient_accumulation_steps: Optional[int] = field(
        default=16, metadata={"help": "the number of gradient accumulation steps"}
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True, metadata={"help": "whether to use gradient checkpointing"}
    )

    eos_padding: Optional[bool] = field(default=True, metadata={"help": "whether to pad with eos token"})
    lora_alpha: Optional[float] = field(default=16, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "the lora dropout parameter"})
    lora_r: Optional[int] = field(default=8, metadata={"help": "the lora r parameter"})

    margin_scale: Optional[float] = field(default=1.0, metadata={"help": "the margin scale"})

    max_prompt_length: Optional[int] = field(default=1000, metadata={"help": "the maximum prompt length"})
    max_length: Optional[int] = field(default=2048, metadata={"help": "the maximum sequence length"})
    max_steps: Optional[int] = field(default=-1, metadata={"help": "max number of training steps"})
    num_train_epochs: Optional[int] = field(default=2, metadata={"help": "max number of training epochs"})
    logging_steps: Optional[int] = field(default=2, metadata={"help": "the logging frequency"})
    save_strategy: Optional[str] = field(default="epoch", metadata={"help": "the saving strategy"})
    save_steps: Optional[int] = field(default=50000, metadata={"help": "the saving frequency"})
    eval_steps: Optional[int] = field(default=100, metadata={"help": "the evaluation frequency"})
    run_name: Optional[str] = field(default="dpo_soft", metadata={"help": "the run name"})
    loss_type: Optional[str] = field(default="sigmoid", metadata={"help": "the loss type"})
    output_dir: Optional[str] = field(default="./dpo_soft", metadata={"help": "the output directory"})
    log_freq: Optional[int] = field(default=1, metadata={"help": "the logging frequency"})

    # instrumentation
    sanity_check: Optional[bool] = field(default=False, metadata={"help": "only train on 1000 samples"})

    max_training_samples: Optional[int] = field(default=-1, metadata={"help": "the maximum sample size"})

    choose_type: Optional[str] = field(default="max_min", metadata={"help": "the choose type"})

    report_to: Optional[str] = field(
        default="none",
        metadata={
            "help": 'The list of integrations to report the results and logs to. Supported platforms are `"azure_ml"`,'
                    '`"comet_ml"`, `"mlflow"`, `"neptune"`, `"tensorboard"`,`"clearml"` and `"wandb"`. '
                    'Use `"all"` to report to all integrations installed, `"none"` for no integrations.'
        },
    )
    # debug argument for distributed training
    ignore_bias_buffers: Optional[bool] = field(
        default=False,
        metadata={
            "help": "fix for DDP issues with LM bias/mask buffers - invalid scalar type,`inplace operation. See"
                    "https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992"
        },
    )
    eot_token: Optional[str] = field(default="", metadata={"help": "the end of text token"})
    mask_prompt: Optional[bool] = field(default=False, metadata={"help": "mask prompt"})
    len_penalty: Optional[float] = field(default=0, metadata={"help": "the length penalty"})
    history_paths: Optional[List[str]] = field(default_factory=list)
    max_history_t: Optional[int] = field(default=2, metadata={"help": "the maximum history length"})


# same function from simpo trainer
def get_batch_logps(
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        average_log_prob: bool = True,
        label_pad_token_id: int = -100,
        is_encoder_decoder: bool = False,
) -> torch.FloatTensor:
    """Compute the log probabilities of the given labels under the given logits.

    Args:
        logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        labels: Labels for which to compute the log probabilities. Label tokens with a value of label_pad_token_id are ignored. Shape: (batch_size, sequence_length)
        average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.
        label_pad_token_id: The label pad token id.
        is_encoder_decoder: Whether the model is an encoder-decoder model.

    Returns:
        A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
    """
    if logits.shape[:-1] != labels.shape:
        raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

    if not is_encoder_decoder:
        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]
    loss_mask = labels != label_pad_token_id

    # dummy token; we'll ignore the losses on these tokens later
    labels[labels == label_pad_token_id] = 0

    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

    if average_log_prob:
        return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
    else:
        return (per_token_logps * loss_mask).sum(-1)


def concatenated_inputs(
        batch: Dict[str, Union[List, torch.LongTensor]],
        padding_value: int = 0,
        label_pad_token_id: int = -100,
) -> Dict[str, torch.LongTensor]:
    """
    接收包含独立 chosen/rejected 张量的批次，并将其拼接。
    """
    concatenated_batch = {}
    max_length = max(batch["chosen_input_ids"].shape[1], batch["rejected_input_ids"].shape[1])

    for k in batch:
        if k.startswith("chosen") and isinstance(batch[k], torch.Tensor):
            pad_value = label_pad_token_id if "labels" in k else padding_value
            concatenated_key = k.replace("chosen", "concatenated")
            concatenated_batch[concatenated_key] = pad_to_length(batch[k], max_length, pad_value=pad_value)

    for k in batch:
        if k.startswith("rejected") and isinstance(batch[k], torch.Tensor):
            pad_value = label_pad_token_id if "labels" in k else padding_value
            concatenated_key = k.replace("rejected", "concatenated")
            concatenated_batch[concatenated_key] = torch.cat(
                (
                    concatenated_batch[concatenated_key],
                    pad_to_length(batch[k], max_length, pad_value=pad_value),
                ),
                dim=0,
            )

    return concatenated_batch


def concatenated_forward(model: nn.Module, batch: Dict) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
    """
    与 DPO Trainer 行为一致的核心计算流程。
    接收一个由 DPODataCollatorWithPadding 处理过的批次，返回 chosen 和 rejected 的 logps。
    """
    # 1. 拼接输入
    concatenated_batch = concatenated_inputs(batch)

    # 2. 准备模型输入
    input_ids = concatenated_batch["concatenated_input_ids"]
    labels = concatenated_batch["concatenated_labels"]
    attention_mask = concatenated_batch["concatenated_attention_mask"]

    # 3. 模型前向传播
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

    # 4. 计算 Logps
    all_logps = get_batch_logps(logits, labels)

    # 5. 分离结果
    bsz = batch["chosen_labels"].shape[0]
    chosen_logps = all_logps[:bsz]
    rejected_logps = all_logps[bsz:]

    return chosen_logps, rejected_logps


def transform_chat_to_str(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    Converts 'chosen' and 'rejected' fields from a list of dicts to a single string.
    It assumes the desired content is in the last message of the list.
    """
    if isinstance(example.get('chosen'), list) and example['chosen']:
        example['chosen'] = example['chosen'][-1]['content']
    if isinstance(example.get('rejected'), list) and example['rejected']:
        example['rejected'] = example['rejected'][-1]['content']
    return example


# a simplified function from simpo trainer
def tokenize_row(feature: Dict[str, Any], tokenizer: PreTrainedTokenizerBase, max_length: int,
                 max_prompt_length: int) -> Dict[str, Any]:
    prompt = feature["prompt"]
    chosen_response = feature["chosen"]
    rejected_response = feature["rejected"]

    prompt_tokens = tokenizer(prompt, max_length=max_prompt_length, truncation=True)
    chosen_tokens = tokenizer(prompt + chosen_response, max_length=max_length, truncation=True)
    rejected_tokens = tokenizer(prompt + rejected_response, max_length=max_length, truncation=True)

    chosen_labels = chosen_tokens["input_ids"][:]
    chosen_labels[:len(prompt_tokens["input_ids"])] = [-100] * len(prompt_tokens["input_ids"])
    rejected_labels = rejected_tokens["input_ids"][:]
    rejected_labels[:len(prompt_tokens["input_ids"])] = [-100] * len(prompt_tokens["input_ids"])

    return {
        "prompt": prompt,  # 保留文本以防万一
        "chosen_input_ids": chosen_tokens["input_ids"],
        "chosen_attention_mask": chosen_tokens["attention_mask"],
        "chosen_labels": chosen_labels,
        "rejected_input_ids": rejected_tokens["input_ids"],
        "rejected_attention_mask": rejected_tokens["attention_mask"],
        "rejected_labels": rejected_labels,
    }


def compute_and_add_logps(
        dataset: DatasetDict, model_path: str, tokenizer: PreTrainedTokenizerBase,
        args: ScriptArguments, accelerator: Accelerator, column_prefix: str
) -> DatasetDict:
    logger.info(f"--- Processing model: {model_path} for columns with prefix: '{column_prefix}' ---")

    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, use_cache=False).eval()
    model = accelerator.prepare_model(model)

    # 核心改动：使用 DPODataCollatorWithPadding
    data_collator = DPODataCollatorWithPadding(
        pad_token_id=tokenizer.pad_token_id,
        label_pad_token_id=-100,
        is_encoder_decoder=False,
    )

    for split in dataset.keys():
        split_dataset = dataset[split]
        # DataLoader 使用新的 collator
        dataloader = DataLoader(
            split_dataset, batch_size=args.per_device_train_batch_size,
            shuffle=False, collate_fn=data_collator
        )
        dataloader = accelerator.prepare(dataloader)

        all_chosen_logps, all_rejected_logps = [], []
        for batch in tqdm(dataloader, desc=f"Computing '{column_prefix}' logps for {split}"):
            with torch.no_grad():
                chosen_logps, rejected_logps = concatenated_forward(model, batch)

            chosen_logps, rejected_logps = accelerator.gather_for_metrics((chosen_logps, rejected_logps))
            all_chosen_logps.append(chosen_logps.cpu())
            all_rejected_logps.append(rejected_logps.cpu())

        # 将 logps 添加回原始的、未被 tokenize 的数据集中
        split_dataset = dataset[split]
        # 第一步：在 split_dataset 上添加 chosen_logps
        split_dataset = split_dataset.add_column(f"{column_prefix}_chosen_logps", torch.cat(all_chosen_logps).numpy())
        # 第二步：在上一步返回的新 dataset 上继续添加 rejected_logps
        split_dataset = split_dataset.add_column(f"{column_prefix}_rejected_logps",
                                                 torch.cat(all_rejected_logps).numpy())
        # 最后，将包含了两个新列的最终结果赋回去
        dataset[split] = split_dataset

    del model
    accelerator.free_memory()
    torch.cuda.empty_cache()
    return dataset


def load_flexible_dataset(dataset_name_or_path, cache_dir=None, split="train"):
    """
    Loads a dataset from a local path (file or directory) or the Hugging Face Hub.

    :param dataset_name_or_path: Path to a file/directory or a Hub dataset name.
    :param cache_dir: Directory for caching data.
    :param split: The dataset split to load.
    :return: The loaded dataset.
    """
    # 检查路径是否指向一个确切存在的文件
    if os.path.isfile(dataset_name_or_path):
        print(f"检测到本地文件路径: {dataset_name_or_path}")
        # 对于单个文件，我们需要指定文件类型。这里我们做得更通用一些。
        file_type = dataset_name_or_path.split('.')[-1]
        if file_type == 'jsonl':
            file_type = 'json'  # .jsonl文件使用json加载器

        print(f"推断文件类型为 '{file_type}'，正在加载...")
        try:
            return load_dataset(
                file_type,
                data_files=dataset_name_or_path,
                split=split,
                cache_dir=cache_dir
            )
        except Exception as e:
            print(f"使用推断的类型 '{file_type}' 加载失败，尝试强制使用 'json' 加载器...")
            # 如果自动推断失败，回退到原始的强制json加载
            return load_dataset(
                "json",
                data_files=dataset_name_or_path,
                split=split,
                cache_dir=cache_dir
            )

    # 如果不是文件，那它可能是目录或Hub名称
    # `load_dataset` 函数本身就能智能处理这两种情况，无需我们手动检查 os.path.isdir
    else:
        if os.path.isdir(dataset_name_or_path):
            print(f"检测到本地文件夹路径: {dataset_name_or_path}，尝试作为已保存的数据集加载...")
            # 1. 从磁盘加载对象，它可能是 Dataset 或 DatasetDict
            loaded_object = load_from_disk(dataset_name_or_path)

            # 2. 检查加载对象的类型
            if isinstance(loaded_object, Dataset):
                # 情况A: 如果加载的是单个Dataset，说明已保存的数据只有一个split。
                # 这种情况下我们直接返回这个Dataset即可。
                print("   -> 加载的对象是单个Dataset，直接返回。")
                return loaded_object

            elif isinstance(loaded_object, DatasetDict):
                # 情况B: 如果加载的是DatasetDict，我们从中选择需要的split。
                # 这是我们之前的逻辑，现在它被放在了正确的位置。
                print("   -> 加载的对象是DatasetDict，从中选择split...")
                if split in loaded_object:
                    return loaded_object[split]
                else:
                    available_splits = list(loaded_object.keys())
                    raise ValueError(
                        f"Split '{split}' not found in the loaded dataset. Available splits are: {available_splits}")

            else:
                # 处理未知类型
                raise TypeError(f"Loaded object from disk is of an unexpected type: {type(loaded_object)}")


        else:
            print(f"未检测到本地路径: {dataset_name_or_path}，尝试从Hugging Face Hub加载...")
            return load_dataset(
                dataset_name_or_path,
                split=split,
                cache_dir=cache_dir
            )


def main():
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    logging.basicConfig(level=logging.INFO)
    accelerator = Accelerator()

    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 1. 加载原始数据集
    logger.info(f"Loading initial raw dataset from: {script_args.train_dir}")
    raw_dataset = load_flexible_dataset(script_args.train_dir)

    logger.info("Transforming 'chosen'/'rejected' columns from list to string...")
    raw_dataset = raw_dataset.map(transform_chat_to_str, num_proc=12)
    logger.info("Transformation complete.")

    if script_args.sanity_check:
        raw_dataset = raw_dataset.select(range(100))

    raw_dataset = DatasetDict({"train": raw_dataset})

    # 2. 对整个数据集进行 Tokenization
    logger.info("Tokenizing dataset in SimPO-style...")
    # 注意：我们在这里不移除原始列，因为 compute_and_add_logps 需要它们
    tokenized_dataset = raw_dataset.map(
        tokenize_row,
        fn_kwargs={
            "tokenizer": tokenizer,
            "max_length": script_args.max_length,
            "max_prompt_length": script_args.max_prompt_length,
        },
        num_proc=12,
    )
    logger.info(f"Dataset tokenized.")

    # 3. 使用 tokenized 数据集进行 logps 计算
    # 注意：我们把 logps 添加回原始的 raw_dataset，而不是 tokenized_dataset
    dataset_with_logps = raw_dataset

    # 计算参考模型的 logps
    dataset_with_logps = compute_and_add_logps(
        dataset=tokenized_dataset, model_path=script_args.ref_model, tokenizer=tokenizer,
        args=script_args, accelerator=accelerator, column_prefix="reference"
    )

    # 遍历历史模型列表
    if script_args.history_paths:
        for i, model_path in enumerate(script_args.history_paths):
            dataset_with_logps = compute_and_add_logps(
                dataset=dataset_with_logps, model_path=model_path, tokenizer=tokenizer,
                args=script_args, accelerator=accelerator, column_prefix=f"history{i}"
            )

    # 保存最终结果
    if accelerator.is_main_process:
        logger.info(f"Saving final dataset to: {script_args.output_dir}")
        dataset_with_logps.save_to_disk(script_args.output_dir)
        logger.info("Script finished successfully.")


if __name__ == "__main__":
    main()


