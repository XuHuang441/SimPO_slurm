import json
from datasets import load_dataset

LOAD_JSON = True

if LOAD_JSON:
    # 假设文件叫 data.json
    with open("/hai/scratch/fangwu97/xu/SimPO_slurm/datasets/gemma2_ultrafeedback/inpo_iter3_20k/all_outputs.json", "r", encoding="utf-8") as f:
        data = json.load(f)
else:
    dataset = load_dataset("princeton-nlp/gemma2-ultrafeedback-armorm")
    data = dataset["train"]

empty_cases = []

empty_strs = []

for i, sample in enumerate(data):
    responses = sample.get("all_generated_responses", [])
    for j, resp in enumerate(responses):
        if isinstance(resp, str) and resp.strip() == "":
            empty_strs.append((i, j))  # i = 第几个样本, j = 第几个response
            print(json.dumps(sample, indent=2, ensure_ascii=False))

    # 遍历 chosen 和 rejected
    for key in ["chosen", "rejected"]:
        for item in sample.get(key, []):
            if item.get("role") == "assistant":
                content = item.get("content", "").strip()
                if content == "":
                    print(json.dumps(sample, indent=2, ensure_ascii=False))
                    empty_cases.append((i, key))  # 记录第几个样本，在哪个key里

# 打印结果
if empty_cases:
    print("发现空的 assistant content：")
    for idx, key in empty_cases:
        print(f"样本 {idx} 的 {key} 里 assistant.content 为空")
else:
    print("没有发现空的 assistant content")

print("-" * 50)

if empty_strs:
    print(f"发现 {len(empty_strs)} 个空字符串：")
    for i, j in empty_strs:
        print(f"样本 {i} 的 all_generated_responses[{j}] 为空")
else:
    print("没有发现空字符串")