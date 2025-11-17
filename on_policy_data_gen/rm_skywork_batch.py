import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import json
import os
from tqdm import tqdm

# --- 1. 配置参数 ---

# 替换为你的模型和数据路径
MODEL_NAME = "Skywork/Skywork-Reward-V2-Llama-3.1-8B"
# 你的GPU服务器上存放模型的缓存目录
CACHE_DIR = "/hai/scratch/fangwu97/xu/cache"
# 输入文件
INPUT_FILE = "/hai/scratch/fangwu97/xu/SimPO_slurm/datasets/gemma2_ultrafeedback/mnpo_iter3_skywork/all_outputs.json"
# 输出文件
OUTPUT_FILE = "/hai/scratch/fangwu97/xu/SimPO_slurm/datasets/gemma2_ultrafeedback/mnpo_iter3_skywork_scored.jsonl"

# 推荐设置一个最大长度以防止OOM，4k对于RM来说通常足够
MAX_SEQ_LENGTH = 4096


# --- 2. 数据加载（使用生成器节省内存） ---
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
        cache_dir=CACHE_DIR
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",  # 自动映射到所有可用GPU (2xH100)
        attn_implementation="flash_attention_2",
        num_labels=1,
        cache_dir=CACHE_DIR
    )
    model.eval()  # 设置为评估模式
    print("         - 模型加载完成.")

    # 检查BOS token，用于后续处理
    bos_token = tokenizer.bos_token
    strip_bos = bos_token is not None
    if strip_bos:
        print(f"         - 将在格式化后移除 BOS token: '{bos_token}'")

    # --- 3.2 准备数据和循环 ---
    print(f"Step 2: 开始处理文件 {INPUT_FILE}...")
    dataset = load_data(INPUT_FILE)

    # 尝试获取总行数用于tqdm进度条
    try:
        total_lines = sum(1 for _ in open(INPUT_FILE, 'r'))
        print(f"         - 共找到 {total_lines} 条样本。")
    except Exception:
        total_lines = 0

    # --- 3.3 循环打分 ---
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        # 使用tqdm显示进度
        for sample in tqdm(dataset, total=total_lines, desc="Scoring responses"):
            prompt = sample['prompt']
            responses = sample['all_generated_responses']

            # 如果没有 response，跳过打分
            if not responses:
                sample['skywork_v2_scores'] = []
                f_out.write(json.dumps(sample, ensure_ascii=False) + '\n')
                continue

            # 1. 为该prompt的所有responses构建一个批次
            conversations_batch = []
            for resp in responses:
                # 构建 [user, assistant] 对话
                conv = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": resp}
                ]
                conversations_batch.append(conv)

            # 2. 批量应用聊天模板
            formatted_batch = tokenizer.apply_chat_template(
                conversations_batch,
                tokenize=False,
            )

            # 3. 批量移除 BOS token (与你的示例逻辑一致)
            if strip_bos:
                formatted_batch_stripped = [
                    s[len(bos_token):] if s.startswith(bos_token) else s
                    for s in formatted_batch
                ]
            else:
                formatted_batch_stripped = formatted_batch

            # 4. 批量 Tokenize
            inputs = tokenizer(
                formatted_batch_stripped,
                return_tensors="pt",
                padding=True,  # 关键：padding到批次中的最大长度
                truncation=True,  # 关键：截断到最大长度
                max_length=MAX_SEQ_LENGTH
            ).to(model.device)  # to(model.device) 确保张量在正确的GPU上

            # 5. 批量推理
            with torch.no_grad():
                logits = model(**inputs).logits
                # logits shape 是 [batch_size, 1]
                # .squeeze(-1) 变为 [batch_size]
                # .float() 从 bfloat16 转为 float (用于 .tolist())
                scores = logits.squeeze(-1).float().cpu().tolist()

            max_idx = scores.index(max(scores))
            min_idx = scores.index(min(scores))

            # 处理边缘情况
            if max_idx == min_idx:
                if len(scores) == 1:
                    # 只有一个 response，chosen 和 rejected 设为同一个
                    chosen_response = responses[0]
                    rejected_response = responses[0]
                else:
                    # 多个 response 但分数全部相同
                    # 按照惯例，取第一个为 chosen，第二个为 rejected
                    chosen_response = responses[0]
                    rejected_response = responses[1]
            else:
                # 正常情况
                chosen_response = responses[max_idx]
                rejected_response = responses[min_idx]

            # --- 8. (新) 覆盖 sample 中的字段 ---

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

            # --- 9. 写入更新后的 sample ---
            f_out.write(json.dumps(sample, ensure_ascii=False) + '\n')

    print(f"\nStep 3: 处理完成！")
    print(f"         - 结果已保存到: {OUTPUT_FILE}")


# --- 4. 运行 ---
if __name__ == "__main__":
    main()