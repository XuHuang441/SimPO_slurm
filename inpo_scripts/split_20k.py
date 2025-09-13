import os
import json
import random

# --- 配置 ---
INPUT_FILE = "/hai/scratch/fangwu97/xu/SimPO_slurm/data/gemma2_ufb_part1.jsonl"
OUTPUT_DIR = "/hai/scratch/fangwu97/xu/SimPO_slurm/data/gemma2_ufb_part1_20k"
NUM_SPLITS = 3
SEED = 42

random.seed(SEED)

# --- 1. 读取 jsonl 文件 ---
print(f"Loading data from {INPUT_FILE}...")
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

print(f"Total samples loaded: {len(data)}")

# --- 2. 打乱顺序 ---
random.shuffle(data)

# --- 3. 切分数据 ---
total = len(data)
split_size = total // NUM_SPLITS
splits = []
for i in range(NUM_SPLITS):
    start = i * split_size
    end = (i + 1) * split_size if i < NUM_SPLITS - 1 else total
    splits.append(data[start:end])
    print(f"Split {i+1}: {len(splits[-1])} samples")

# --- 4. 保存为 jsonl ---
for i, split in enumerate(splits, start=1):
    out_file = os.path.join(OUTPUT_DIR, f"gemma2_ufb_part1_split{i}.jsonl")
    with open(out_file, "w", encoding="utf-8") as f:
        for item in split:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Saved split {i} -> {out_file}")

print("\nAll splits saved successfully!")
