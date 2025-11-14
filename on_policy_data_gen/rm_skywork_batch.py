# Add batch processing to `reward_model_annotate_skywork.py` to better leverage the GPU and accelerate the pipeline.import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import json
import os
import argparse
import tqdm
import numpy as np
import datasets
import torch

parser = argparse.ArgumentParser()
parser.add_argument(
    "--generation_file",
    type=str,
    default="datasets/gemma2_ultrafeedback/all_outputs.json",
    help="Path to the output generation file",
)
parser.add_argument(
    "--reward_model",
    type=str,
    default="Skywork/Skywork-Reward-V2-Llama-3.1-8B",
    help="Path to reward model",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="datasets/gemma2_ultrafeedback/",
    help="Path to output directory",
)
parser.add_argument(
    "--cache_dir",
    type=str,
    default=None,
    help="Cache directory for model and dataset",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=64,
    help="Batch size for reward model inference",
)
args = parser.parse_args()
print(args)

generation_file = args.generation_file

# 读取 JSONL
with open(generation_file, "r") as f:
    output_data = [json.loads(line) for line in f]

# -------------------------------------------------
# 1. 加载模型 & tokenizer
# -------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModelForSequenceClassification.from_pretrained(
    args.reward_model,
    device_map=device,          # 8B 模型放在一张卡上就够了
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    cache_dir=args.cache_dir,
    attn_implementation="flash_attention_2",
    num_labels=1,
)
tokenizer = AutoTokenizer.from_pretrained(args.reward_model, cache_dir=args.cache_dir)

model.eval()

# -------------------------------------------------
# 2. 构建“全局样本列表”，每个元素是一个 (data_idx, cand_idx, prompt, candidate)
#    然后按 batch_size 切块，一次前向算一大批
# -------------------------------------------------
examples = []  # 每个元素: (data_idx, cand_idx, prompt, candidate_text)
for i, data in enumerate(output_data):
    prompt = data["prompt"]
    for j, cand in enumerate(data["all_generated_responses"]):
        examples.append((i, j, prompt, cand))

print(f"Total (prompt, candidate) pairs: {len(examples)}")

# 用一个二维 list 存所有分数，之后再回填
all_scores = [
    [None] * len(d["all_generated_responses"]) for d in output_data
]

batch_size = args.batch_size

# -------------------------------------------------
# 3. Batch 化打分
# -------------------------------------------------
for start in tqdm.tqdm(range(0, len(examples), batch_size), desc="Scoring with RM"):
    end = min(start + batch_size, len(examples))
    batch = examples[start:end]

    # 3.1 构造 batch 的输入文本（先用 chat_template 变成字符串）
    formatted_inputs = []
    index_pairs = []  # (data_idx, cand_idx) 用来之后回填 scores
    for data_idx, cand_idx, prompt, candidate in batch:
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": candidate},
        ]
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
        )

        # 防止重复 bos token（有些模型会这样建议）
        if tokenizer.bos_token is not None and formatted.startswith(tokenizer.bos_token):
            formatted = formatted[len(tokenizer.bos_token):]

        formatted_inputs.append(formatted)
        index_pairs.append((data_idx, cand_idx))

    # 3.2 一次性 tokenization + padding
    enc = tokenizer(
        formatted_inputs,
        padding=True,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    # 3.3 前向推理，取 logits 作为 reward
    with torch.no_grad():
        outputs = model(**enc)   # logits: [B, 1]
        logits = outputs.logits.squeeze(-1).float().cpu().tolist()  # [B]

    # 3.4 将 scores 写回 all_scores
    for (data_idx, cand_idx), score in zip(index_pairs, logits):
        all_scores[data_idx][cand_idx] = float(score)

# 检查是否有 None 遗漏
for i, row_scores in enumerate(all_scores):
    if any(s is None for s in row_scores):
        raise ValueError(f"Missing score in sample {i}")

# 将分数写回原数据
for i, data in enumerate(output_data):
    data["all_rm_scores"] = all_scores[i]

# -------------------------------------------------
# 4. 保存带 rm 分数的文件：*_rm.json
# -------------------------------------------------
file_name = os.path.basename(args.generation_file).split(".json")[0] + "_rm.json"
os.makedirs(args.output_dir, exist_ok=True)
rm_path = os.path.join(args.output_dir, file_name)
with open(rm_path, "w") as f:
    json.dump(output_data, f, indent=4, ensure_ascii=False)

print(f"Annotated outputs saved to {rm_path}")

# -------------------------------------------------
# 5. Binarize：选分数最高为 chosen，最低为 rejected
# -------------------------------------------------
for data in output_data:
    scores = data["all_rm_scores"]
    chosen_idx = int(np.argmax(scores))
    rejected_idx = int(np.argmin(scores))

    chosen = [
        {"role": "user", "content": data["prompt"]},
        {"role": "assistant", "content": data["all_generated_responses"][chosen_idx]},
    ]
    rejected = [
        {"role": "user", "content": data["prompt"]},
        {"role": "assistant", "content": data["all_generated_responses"][rejected_idx]},
    ]

    data.update(
        {
            "chosen": chosen,
            "rejected": rejected,
        }
    )

# 保存 *_bin.json
output_file = os.path.basename(args.generation_file).split(".json")[0] + "_bin.json"
bin_path = os.path.join(args.output_dir, output_file)
with open(bin_path, "w") as f:
    json.dump(output_data, f, indent=4, ensure_ascii=False)
print(f"Binarized outputs saved to {bin_path}")

# -------------------------------------------------
# 6. 转成 HF Dataset，保存到磁盘
# -------------------------------------------------
dataset_output_path = os.path.join(
    args.output_dir,
    os.path.basename(args.generation_file).split(".json")[0] + "_bin_dataset",
)
dataset = datasets.Dataset.from_list(output_data)
dataset.save_to_disk(dataset_output_path)
print(f"Binarized dataset saved to {dataset_output_path}")
