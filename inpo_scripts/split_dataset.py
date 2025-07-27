from datasets import load_dataset

# 1. 加载完整数据集
print("Loading the full dataset...")
full_dataset = load_dataset("princeton-nlp/gemma2-ultrafeedback-armorm", split="train") # 只加载 train split

# 2. 固定种子并打乱数据集
print("Shuffling dataset with a fixed seed...")
shuffled_dataset = full_dataset.shuffle(seed=42) # 使用你的种子

# 3. 计算分割点
total_rows = len(shuffled_dataset)
split_size = total_rows // 3
print(f"Total rows: {total_rows}, each split will have approx. {split_size} rows.")

# 4. 分割成3个部分
dataset_part1 = shuffled_dataset.select(range(split_size))
dataset_part2 = shuffled_dataset.select(range(split_size, 2 * split_size))
dataset_part3 = shuffled_dataset.select(range(2 * split_size, total_rows)) # 最后一个部分包含余数

print(f"Part 1 size: {len(dataset_part1)}")
print(f"Part 2 size: {len(dataset_part2)}")
print(f"Part 3 size: {len(dataset_part3)}")

# 5. 保存到磁盘，为你的3轮迭代做准备
print("Saving splits to disk...")
dataset_part1.to_json("../data/gemma2_ufb_part1.jsonl")
dataset_part2.to_json("../data/gemma2_ufb_part2.jsonl")
dataset_part3.to_json("../data/gemma2_ufb_part3.jsonl")

print("Done!")

# 在你的工作流中：
# - 迭代1 使用 gemma2_ufb_part1.jsonl 进行预计算和训练
# - 迭代2 使用 gemma2_ufb_part2.jsonl 进行预计算和训练
# - 迭代3 使用 gemma2_ufb_part3.jsonl 进行预计算和训练