import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import json
import os
from tqdm import tqdm

# --- 1. 配置参数 ---

# 替换为你的模型和数据路径
MODEL_NAME = "RLHFlow/ArmoRM-Llama3-8B-v0.1"
# 你的GPU服务器上存放模型的缓存目录
CACHE_DIR = "/hai/scratch/fangwu97/xu/cache"  # 你要求的 cachedir
# 输入文件
INPUT_FILE = "/hai/scratch/fangwu97/xu/SimPO_slurm/datasets/gemma2_ultrafeedback/mnpo_iter3_armo/all_outputs.json"
# 输出文件 (DPO-ready format)
OUTPUT_FILE = "/hai/scratch/fangwu97/xu/SimPO_slurm/datasets/gemma2_ultrafeedback/mnpo_iter3_armo_scored.jsonl"

# ArmoRM 示例中使用的最大长度
MAX_SEQ_LENGTH = 4096


def load_data(file_path):
    suffix = os.path.splitext(file_path)[1].lower()

    # JSONL: 一行一个 JSON 对象
    if suffix in [".jsonl", ".jsonlines", ".ljson"]:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)

    # JSON: 整个文件可能是 list 或 dict
    elif suffix == ".json":
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 如果是 list → 逐个 yield
        if isinstance(data, list):
            for item in data:
                yield item

        # 如果是 dict → 只 yield 一次
        elif isinstance(data, dict):
            yield data

        else:
            raise ValueError("JSON 文件顶层必须是 list 或 dict")

    else:
        raise ValueError(f"Unsupported file format: {suffix}")


# --- 3. 主函数 ---
def main():
    # --- 3.1 加载模型和Tokenizer ---
    print(f"Step 1: 正在加载模型 {MODEL_NAME}...")
    print(f"         - Cache Dir: {CACHE_DIR}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        cache_dir=CACHE_DIR,
        use_fast=True  # 根据 ArmoRM 示例
    )

    # 注意: ArmoRM 需要 trust_remote_code=True
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,  # must enable for armorm
        cache_dir=CACHE_DIR
    )
    model.eval()  # 设置为评估模式
    print("         - 模型加载完成.")

    # --- 3.2 准备数据和循环 ---
    print(f"Step 2: 开始处理文件 {INPUT_FILE}...")
    dataset = load_data(INPUT_FILE)

    # 尝试获取总行数用于tqdm进度条
    try:
        total_lines = sum(1 for _ in open(INPUT_FILE, 'r'))
        print(f"         - 共找到 {total_lines} 条样本。")
    except Exception:
        total_lines = 0

    # --- 3.3 循环打分并格式化 ---
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        # 使用tqdm显示进度
        for sample in tqdm(dataset, total=total_lines, desc="Scoring and creating pairs"):
            prompt = sample['prompt']
            responses = sample['all_generated_responses']

            # 1. 处理没有 responses 的情况
            if not responses:
                sample['all_rm_scores'] = []
                sample['chosen'] = []
                sample['rejected'] = []
                f_out.write(json.dumps(sample, ensure_ascii=False) + '\n')
                continue

            # 2. 为该prompt的所有responses构建一个批次
            # apply_chat_template 可以批量处理 list[list[dict]]
            conversations_batch = []
            for resp in responses:
                conv = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": resp}
                ]
                conversations_batch.append(conv)

            # 3. 批量 Tokenize (与 ArmoRM 示例一致)
            # 这种方式更高效，直接一步到位
            input_ids = tokenizer.apply_chat_template(
                conversations_batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_SEQ_LENGTH,
            ).to("cuda")

            # 4. 批量推理，获取分数
            with torch.no_grad():
                output = model(input_ids)
                # 分数在 output.score 属性中, shape: [batch_size]
                scores = output.score.float().cpu().tolist()

            # --- 5. 根据分数确定 Chosen 和 Rejected ---

            max_idx = scores.index(max(scores))
            min_idx = scores.index(min(scores))

            # 处理边缘情况
            if max_idx == min_idx:
                if len(scores) == 1:
                    chosen_response = responses[0]
                    rejected_response = responses[0]
                else:
                    # 多个 response 但分数全部相同
                    chosen_response = responses[0]
                    rejected_response = responses[1]
            else:
                chosen_response = responses[max_idx]
                rejected_response = responses[min_idx]

            # --- 6. 覆盖 sample 中的字段 ---

            # 覆盖分数字段
            sample['all_rm_scores'] = scores

            # 覆盖 chosen 字段
            sample['chosen'] = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": chosen_response}
            ]

            # 覆盖 rejected 字段
            sample['rejected'] = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": rejected_response}
            ]

            # --- 7. 写入更新后的 sample ---
            f_out.write(json.dumps(sample, ensure_ascii=False) + '\n')

    print(f"\nStep 3: 处理完成！")
    print(f"         - ArmoRM DPO 格式数据已保存到: {OUTPUT_FILE}")


# --- 4. 运行 ---
if __name__ == "__main__":
    main()