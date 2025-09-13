from datasets import load_from_disk, DatasetDict
import os

# --- 配置 ---
DATASET_PATH = "/hai/scratch/fangwu97/xu/SimPO_slurm/data/gemma2_ufb_part1"
SEED = 42
NUM_SPLITS = 3
BASE_SAVE_PATH = "/hai/scratch/fangwu97/xu/SimPO_slurm/data/gemma2_ufb_part1_20k"
JSON_SAVE_PATH = "/hai/scratch/fangwu97/xu/SimPO_slurm/data/gemma2_ufb_part1_20k"

# --- 加载本地数据集 ---
print(f"Loading local dataset from '{DATASET_PATH}'...")
full_dataset_dict = load_from_disk(DATASET_PATH)
print(f"Dataset loaded. Available splits: {list(full_dataset_dict.keys())}")

train_dataset = full_dataset_dict['train']
test_dataset = full_dataset_dict['test']


# --- 1. 定义一个函数来分割 train split ---
def split_train(dataset, num_splits, seed):
    print(f"\nShuffling and splitting train data with seed {seed}...")
    shuffled_dataset = dataset.shuffle(seed=seed)

    total_rows = len(shuffled_dataset)
    split_size = total_rows // num_splits
    print(f"Total rows in train: {total_rows}, each of the {num_splits} splits will have approx. {split_size} rows.")

    splits = []
    for i in range(num_splits):
        start_index = i * split_size
        end_index = (i + 1) * split_size if i < num_splits - 1 else total_rows
        splits.append(shuffled_dataset.select(range(start_index, end_index)))

    return splits


# --- 2. 拆分 train，保持 test 不变 ---
train_parts = split_train(train_dataset, NUM_SPLITS, SEED)

# --- 3. 保存每个部分 ---
for i in range(NUM_SPLITS):
    part_num = i + 1
    print(f"\n--- Processing and saving Part {part_num} ---")

    # 每个部分用 train 的一份 + 原始的 test
    final_dataset_part = DatasetDict({
        'train': train_parts[i],
        'test': test_dataset
    })

    print(f"Part {part_num} train size: {len(final_dataset_part['train'])}")
    print(f"Part {part_num} test size: {len(final_dataset_part['test'])}")

    train_json_path = os.path.join(JSON_SAVE_PATH, f"gemma2_ufb_part{part_num}_train.jsonl")
    final_dataset_part['train'].to_json(train_json_path)
    print(f"Saved JSONL files to: {train_json_path}")

    # 保存到本地
    disk_path = os.path.join(BASE_SAVE_PATH, f"gemma2_ufb_part1_split{part_num}")
    final_dataset_part.save_to_disk(disk_path)
    print(f"Saved dataset part {part_num} to disk at: {disk_path}")

print("\nAll train splits processed and saved successfully!")

# --- 使用方法 ---
# from datasets import load_from_disk
# dataset_iter1 = load_from_disk("/hai/scratch/fangwu97/xu/SimPO_slurm/data/gemma2_ufb_part1_split1")
# train_data_1 = dataset_iter1['train']
# test_data_1 = dataset_iter1['test']
