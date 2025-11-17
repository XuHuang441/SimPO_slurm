import json
import torch
from torch import nn
from transformers import (
    LlamaModel,
    LlamaPreTrainedModel,
    TextClassificationPipeline,
    AutoTokenizer,
    pipeline
)
from typing import Dict, List, Any, Generator
from tqdm import tqdm
import os

# ================= 配置区域 =================
INPUT_FILE = "/hai/scratch/fangwu97/xu/SimPO_slurm/data/gemma2_ufb_part1_20k/gemma2_ufb_part1_split1.jsonl"
OUTPUT_FILE = "/hai/scratch/fangwu97/xu/SimPO_slurm/datasets/gemma2_ultrafeedback/mnpo_iter1_athene_scored.jsonl"
# 请将其替换为你实际的缓存路径
CACHE_DIR = "/hai/scratch/fangwu97/xu/cache"
MODEL_NAME = "Nexusflow/Athene-RM-8B"

# 确保缓存目录存在
os.makedirs(CACHE_DIR, exist_ok=True)


# ================= 模型定义 (保持你提供的代码一致) =================

class AtheneForSequenceClassification(LlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.v_head = nn.Linear(config.hidden_size, 1, bias=False)
        self.CLS_ID = 128003
        # Initialize weights and apply final processing
        self.post_init()

    def get_device(self):
        return self.model.device

    def forward(
            self,
            input_ids=None,
            past_key_values=None,
            attention_mask=None,
            position_ids=None,
    ):
        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
        )
        hidden_states = transformer_outputs.hidden_states[-1]
        rewards = self.v_head(hidden_states).squeeze(-1)

        bs = int(input_ids.shape[0])
        scores = []
        for i in range(bs):
            # 找到 CLS token 的位置
            c_inds = (input_ids[i] == self.CLS_ID).nonzero()
            if c_inds.numel() > 0:
                c_ind = c_inds[-1].item()
                scores.append(rewards[i, c_ind])
            else:
                # Fallback if CLS token not found (rare case handling)
                scores.append(rewards[i, -1])

        scores = torch.stack(scores)
        return {"scores": scores}


class AtheneRewardPipeline(TextClassificationPipeline):
    def preprocess(self, inputs, **tokenizer_kwargs) -> Dict[str, torch.Tensor]:
        return_tensors = self.framework
        # 应用 chat template
        formatted = self.tokenizer.apply_chat_template(inputs, tokenize=False)
        formatted = formatted + self.tokenizer.cls_token

        return self.tokenizer(
            formatted,
            return_tensors=return_tensors,
            max_length=4096,
            padding="longest",
            truncation=True,
        )

    def postprocess(self, model_outputs, function_to_apply=None, top_k=1, _legacy=True):
        return model_outputs["scores"].cpu().float().item()


def load_data_generator(file_path: str) -> tuple[Generator[Dict, None, None], int]:
    """
    读取数据，兼容 json 和 jsonl。
    返回: (数据生成器, 数据总条数)
    """
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    if ext == '.json':
        print(f"Detected JSON format: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    return iter(data), len(data)
                else:
                    # 只有一个对象的情况
                    return iter([data]), 1
        except json.JSONDecodeError:
            # 如果扩展名是json但内容是jsonl，尝试回退到按行读取
            print("Warning: Failed to parse as JSON list, trying JSONL mode...")
            return _load_jsonl(file_path)

    else:  # 默认为 jsonl
        print(f"Detected JSONL format: {file_path}")
        return _load_jsonl(file_path)


def _load_jsonl(file_path):
    # 先扫一遍文件计算行数用于 tqdm
    print("Counting lines for progress bar...")
    with open(file_path, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for line in f if line.strip())

    def generator():
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue

    return generator(), total_lines

# ================= 主处理逻辑 =================

def main():
    print(f"Loading model: {MODEL_NAME}...")

    # 1. 加载 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        cache_dir=CACHE_DIR,
        trust_remote_code=True
    )

    # 2. 加载 Model (使用 bfloat16 以适配 H100 性能)
    model = AtheneForSequenceClassification.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",  # 自动分配到 GPU
        cache_dir=CACHE_DIR,
        trust_remote_code=True
    )

    # 3. 初始化 Pipeline
    reward_pipe = pipeline(
        task="text-classification",
        model=model,
        tokenizer=tokenizer,
        pipeline_class=AtheneRewardPipeline,
        device_map="auto",
    )

    print("Starting processing...")

    data_iter, total_count = load_data_generator(INPUT_FILE)

    # 以 append 模式写入，如果是重新运行建议先手动删除旧文件
    if os.path.exists(OUTPUT_FILE):
        print(f"Warning: Output file {OUTPUT_FILE} exists. Appending to it.")

        with open(OUTPUT_FILE, 'a', encoding='utf-8') as fout:

            for data in tqdm(data_iter, total=total_count, desc="Scoring"):
                prompt = data.get('prompt', "")
                responses = data.get('all_generated_responses', [])

                if not responses:
                    # 如果没有回复，直接写回原数据（可选）
                    fout.write(json.dumps(data, ensure_ascii=False) + '\n')
                    continue

                # 构建 pipeline 输入
                pipe_inputs = []
                for resp in responses:
                    pipe_inputs.append([
                        {"role": 'user', "content": prompt},
                        {"role": "assistant", "content": resp}
                    ])

                # 批量推理
                try:
                    scores = reward_pipe(pipe_inputs, batch_size=len(pipe_inputs))
                except Exception as e:
                    print(f"\nError processing prompt_id {data.get('prompt_id')}: {e}")
                    continue

                data['all_rm_scores'] = scores

                # 找到 chosen / rejected
                max_score_idx = scores.index(max(scores))
                min_score_idx = scores.index(min(scores))

                data['chosen'] = pipe_inputs[max_score_idx]
                data['rejected'] = pipe_inputs[min_score_idx]

                # 写入一行 (JSONL格式)
                fout.write(json.dumps(data, ensure_ascii=False) + '\n')

    print(f"\nProcessing complete. Output saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()