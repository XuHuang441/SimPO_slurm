import os
import json
from dataclasses import dataclass, field
from typing import Optional, List

import numpy as np
import torch

from datasets import Dataset, load_dataset, load_from_disk, DatasetDict
from inpo_scripts.precompute_trainer import PreComputer
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageClassification,
    AutoTokenizer,
    HfArgumentParser,
)
from tqdm import tqdm
from trl import DPOConfig 

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
            file_type = 'json' # .jsonl文件使用json加载器
        
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
                    raise ValueError(f"Split '{split}' not found in the loaded dataset. Available splits are: {available_splits}")
            
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

def prepare_data(
    dataset_name_or_path: str = "princeton-nlp/gemma2-ultrafeedback-armorm",
    sanity_check: bool = False,
    cache_dir: str = None,
    num_proc=24,
    eot_token="",
    length_penalty=0, # 此参数在数据准备阶段未使用，但保留以保持函数签名一致
) -> Dataset:
    """
    为DPO训练准备数据集，此函数适配 "princeton-nlp/gemma2-ultrafeedback-armorm" 格式。
    它会直接从Hugging Face Hub加载数据集，并提取 prompt, chosen, 和 rejected 列。
    """
    print(f"从 '{dataset_name_or_path}' 加载数据...")

    ds = load_flexible_dataset(dataset_name_or_path,cache_dir=cache_dir)

    # 如果是健全性检查，只选择少量样本
    if sanity_check:
        print("执行健全性检查，仅使用100个样本。")
        ds = ds.select(range(min(len(ds), 100)))

    prompts = []
    pos = []  # 存储 "chosen" 回答
    neg = []  # 存储 "rejected" 回答

    # 遍历数据集并提取所需字段
    # 使用tqdm来显示进度条
    for sample in tqdm(ds, desc="正在处理数据集"):
        # prompt 是一个顶层字段，直接获取
        prompts.append(sample["prompt"])

        # 'chosen' 和 'rejected' 字段是一个包含两个字典的列表。
        # 第一个字典是用户角色，第二个字典是助手的回答。我们需要的是助手的回答内容。
        if sample.get("chosen") and len(sample["chosen"]) > 1 and sample["chosen"][1].get("role") == "assistant":
            chosen_content = sample["chosen"][1]["content"]
            pos.append(chosen_content + eot_token)
        else:
            # 如果格式不符，我们可以跳过这个样本，但需要确保列表长度一致
            # 为了简单起见，这里假设数据格式总是正确的
            continue

        if sample.get("rejected") and len(sample["rejected"]) > 1 and sample["rejected"][1].get("role") == "assistant":
            rejected_content = sample["rejected"][1]["content"]
            neg.append(rejected_content + eot_token)
        else:
            # 如果rejected格式不符，为了保持数据对齐，我们需要移除刚刚添加的prompt和chosen
            prompts.pop()
            pos.pop()
            continue

    # 确保所有列表的长度都相等，这是一个好习惯
    assert len(prompts) == len(pos) == len(neg), "处理后的样本数量不匹配！"

    # 用提取出的数据创建一个新的Dataset对象
    dataset = Dataset.from_dict({"prompt": prompts, "chosen": pos, "rejected": neg})
    
    print(f"成功加载并处理了 {len(dataset)} 条样本。")

    return dataset

def precompute_multi_history(
    history_model_paths: List[str],
):
    history_logps_list = []
    for step_idx, model_path in enumerate(history_model_paths):
        print(f"[History Step {step_idx}] Loading model from: {model_path}")

        history_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            use_flash_attention_2=True,
            use_cache=False
        )

        pre = PreComputer(
            random_model,
            ref_model=history_model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            mask_prompt=script_args.mask_prompt,
            len_penalty=script_args.len_penalty,
        )

        chosen_logps, rejected_logps = pre.precompute()
        history_logps_list.append((chosen_logps, rejected_logps))
    return history_logps_list


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    # model = AutoModelForCausalLM.from_pretrained(
    #     script_args.model_name_or_path,
    #     use_flash_attention_2=True,
    #     torch_dtype=torch.float16,
    # )
    # model.config.use_cache = False

    random_model = AutoModelForCausalLM.from_pretrained(script_args.model_name_or_path, use_cache=False)

    if script_args.ignore_bias_buffers:
        # torch distributed hack
        random_model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in random_model.named_buffers() if buffer.dtype == torch.bool
        ]

    if script_args.ref_model:
        ref_name = script_args.ref_model
    else:
        ref_name = script_args.model_name_or_path

    last_name = script_args.last_model

    model = AutoModelForCausalLM.from_pretrained(
        ref_name,
        torch_dtype=torch.bfloat16,
        use_flash_attention_2=True,
        use_cache=False
    )
    

    tokenizer = AutoTokenizer.from_pretrained(ref_name)
    tokenizer.padding_side = 'left'

    if script_args.eos_padding:
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        model.config.vocab_size += 1
        ref_model.config.vocab_size += 1
        model.config.pad_token_id = tokenizer.pad_token_id
        ref_model.config.pad_token_id = tokenizer.pad_token_id
        model.resize_token_embeddings(len(tokenizer))
        ref_model.resize_token_embeddings(len(tokenizer))

    # 2. Load the Stack-exchange paired dataset
    train_dataset = prepare_data(
        dataset_name_or_path=script_args.train_dir,
        sanity_check=script_args.sanity_check,
        eot_token=script_args.eot_token,
        length_penalty=script_args.len_penalty,
    )
    if script_args.max_training_samples > 0:
        train_dataset = train_dataset.select(range(script_args.max_training_samples))

    training_args = DPOConfig(
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        per_device_eval_batch_size=script_args.per_device_eval_batch_size,
        # max_steps=script_args.max_steps,
        num_train_epochs=script_args.num_train_epochs,
        save_strategy=script_args.save_strategy,
        logging_steps=script_args.logging_steps,
        save_steps=script_args.save_steps,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        gradient_checkpointing=script_args.gradient_checkpointing,
        learning_rate=script_args.learning_rate,
        output_dir=script_args.output_dir,
        # report_to=script_args.report_to,
        lr_scheduler_type=script_args.lr_scheduler_type,
        warmup_steps=script_args.warmup_steps,
        # optim=script_args.optimizer_type,
        bf16=True,
        remove_unused_columns=False,
        run_name=script_args.run_name,
        beta=script_args.beta,
        loss_type=script_args.loss_type,
        max_prompt_length=script_args.max_prompt_length,
        max_length=script_args.max_length,
    )
    # print(training_args)

    # 5. initialize the DPO trainer

    pre = PreComputer(
        random_model,
        ref_model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        mask_prompt=script_args.mask_prompt,
        len_penalty=script_args.len_penalty,
    )
    print("begin to precompute")
    reference_chosen_logps, reference_rejected_logps = pre.precompute()
    # for s in pre_dataset:
    #     print(len(s["chosen_input_ids"]), len(s["chosen_attention_mask"]), len(s["chosen_labels"]))

    # precompute history model logps
    history_paths = script_args.history_paths
    if script_args.max_history_t > 0 and history_paths:
        history_paths = history_paths[-script_args.max_history_t:][::-1]

    history_logps = []
    if not history_paths: # iter = 1
        history_logps = precompute_multi_history(
            history_model_paths=[script_args.last_model],
        )
    else:
        history_logps = precompute_multi_history(
            history_model_paths=history_paths,
        )

    pre_dataset = pre.train_dataset

    pre_dataset = pre_dataset.add_column(name="reference_chosen_logps", column=reference_chosen_logps)
    pre_dataset = pre_dataset.add_column(name="reference_rejected_logps", column=reference_rejected_logps)

    for j, (cj, rj) in enumerate(history_logps):
        pre_dataset = pre_dataset.add_column(f"history{j}_chosen_logps", cj)
        pre_dataset = pre_dataset.add_column(f"history{j}_rejected_logps", rj)

    pre_dataset.save_to_disk(script_args.output_dir, num_shards=1)
    # with open(output_path, "w", encoding="utf8") as f:
    #     json.dump(pre_dataset, f, ensure_ascii=False)


   
   