from datasets import load_dataset, DatasetDict
import os

# --- 配置 ---
DATASET_NAME = "princeton-nlp/gemma2-ultrafeedback-armorm"
SEED = 42
NUM_SPLITS = 3
BASE_SAVE_PATH = "/hai/scratch/fangwu97/xu/SimPO_slurm/data"
JSON_SAVE_PATH = "/hai/scratch/fangwu97/xu/SimPO_slurm/data"

# --- 确保保存目录存在 ---
os.makedirs(BASE_SAVE_PATH, exist_ok=True)
os.makedirs(JSON_SAVE_PATH, exist_ok=True)

# --- 1. 加载包含所有 splits 的完整数据集 ---
print(f"Loading the full dataset '{DATASET_NAME}'...")
full_dataset_dict = load_dataset(DATASET_NAME)
print(f"Dataset loaded. Available splits: {list(full_dataset_dict.keys())}")


# --- 2. 定义一个函数来分割单个 split (例如 'train' 或 'test') ---
def split_data(dataset, num_splits, seed):
    """将给定的 dataset 对象分割成指定数量的部分"""
    print(f"\nShuffling and splitting data with seed {seed}...")
    shuffled_dataset = dataset.shuffle(seed=seed)

    total_rows = len(shuffled_dataset)
    split_size = total_rows // num_splits
    print(f"Total rows: {total_rows}, each of the {num_splits} splits will have approx. {split_size} rows.")

    splits = []
    for i in range(num_splits):
        start_index = i * split_size
        # 对于最后一个 split，确保它包含所有剩余的行
        end_index = (i + 1) * split_size if i < num_splits - 1 else total_rows
        splits.append(shuffled_dataset.select(range(start_index, end_index)))

    return splits


# --- 3. 分别处理 'train' 和 'test' splits ---
train_parts = split_data(full_dataset_dict['train'], NUM_SPLITS, SEED)
test_parts = split_data(full_dataset_dict['test'], NUM_SPLITS, SEED)

# --- 4. 组合并保存每个部分 ---
for i in range(NUM_SPLITS):
    part_num = i + 1
    print(f"\n--- Processing and saving Part {part_num} ---")

    # 组合 train 和 test parts 来创建一个新的 DatasetDict
    final_dataset_part = DatasetDict({
        'train': train_parts[i],
        'test': test_parts[i]
    })

    print(f"Part {part_num} train size: {len(final_dataset_part['train'])}")
    print(f"Part {part_num} test size: {len(final_dataset_part['test'])}")

    # a. 保存为 JSONL 文件
    train_json_path = os.path.join(JSON_SAVE_PATH, f"gemma2_ufb_part{part_num}_train.jsonl")
    test_json_path = os.path.join(JSON_SAVE_PATH, f"gemma2_ufb_part{part_num}_test.jsonl")

    final_dataset_part['train'].to_json(train_json_path)
    final_dataset_part['test'].to_json(test_json_path)
    print(f"Saved JSONL files to: {train_json_path} and {test_json_path}")

    # b. 使用 save_to_disk 保存为 huggingface datasets 格式
    disk_path = os.path.join(BASE_SAVE_PATH, f"gemma2_ufb_part{part_num}")
    final_dataset_part.save_to_disk(disk_path)
    print(f"Saved dataset part {part_num} to disk at: {disk_path}")

print("\nAll parts processed and saved successfully!")

# --- 如何在工作流中使用 ---
# 在你的训练脚本中，你可以这样加载每个部分：
# from datasets import load_from_disk
#
# # 迭代1
# dataset_iter1 = load_from_disk("/home/ubuntu/xu/SimPO/data/gemma2_ufb_part1")
# train_data_1 = dataset_iter1['train']
# test_data_1 = dataset_iter1['test']
#
# # 迭代2
# dataset_iter2 = load_from_disk("/home/ubuntu/xu/SimPO/data/gemma2_ufb_part2")
# ...以此类推